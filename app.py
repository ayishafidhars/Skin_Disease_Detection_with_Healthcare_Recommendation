"""
AI-Based Intelligent System for Skin Disease Detection
Streamlit Web Application
"""

import streamlit as st
import os
import sys
import json
import sqlite3
import hashlib
import hmac
import secrets
import smtplib
from email.message import EmailMessage
from urllib.parse import urlencode
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import prediction functions
from predict import predict_skin_disease, get_health_recommendation, load_class_names

AUTH_DB_PATH = os.path.join(SCRIPT_DIR, 'users.db')
PBKDF2_ITERATIONS = 200_000
RESET_TOKEN_TTL_MINUTES = 30
GOOGLE_AUTH_URL = 'https://accounts.google.com/o/oauth2/v2/auth'
GOOGLE_TOKEN_URL = 'https://oauth2.googleapis.com/token'


def _get_streamlit_secret_value(*path_parts):
    try:
        current = st.secrets
        for part in path_parts:
            if part not in current:
                return ''
            current = current[part]
        return str(current).strip() if current is not None else ''
    except Exception:
        return ''


def _get_config_value(env_var_name, *secret_path_parts):
    env_value = os.getenv(env_var_name, '').strip()
    if env_value:
        return env_value
    return _get_streamlit_secret_value(*secret_path_parts)


def get_db_connection():
    connection = sqlite3.connect(AUTH_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_auth_db():
    connection = get_db_connection()
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                phone_number TEXT,
                age INTEGER,
                gender TEXT,
                section TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Backward-compatible schema migration for existing DBs.
        existing_columns = {
            row['name'] for row in connection.execute("PRAGMA table_info(users)")
        }
        if 'phone_number' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN phone_number TEXT")
        if 'age' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN age INTEGER")
        if 'gender' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN gender TEXT")
        if 'section' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN section TEXT")
        if 'oauth_provider' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN oauth_provider TEXT")
        if 'oauth_sub' not in existing_columns:
            connection.execute("ALTER TABLE users ADD COLUMN oauth_sub TEXT")

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_phone_unique
            ON users(phone_number)
            WHERE phone_number IS NOT NULL AND phone_number <> ''
            """
        )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub_unique
            ON users(oauth_provider, oauth_sub)
            WHERE oauth_provider IS NOT NULL AND oauth_sub IS NOT NULL
            """
        )

        connection.commit()
    finally:
        connection.close()


def hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    password_bytes = password.encode('utf-8')
    digest = hashlib.pbkdf2_hmac('sha256', password_bytes, salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        algorithm, iterations_str, salt_hex, hash_hex = stored_hash.split('$', 3)
        if algorithm != 'pbkdf2_sha256':
            return False
        iterations = int(iterations_str)
        candidate = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            bytes.fromhex(salt_hex),
            iterations,
        )
        return hmac.compare_digest(candidate.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


def validate_password_strength(password):
    if len(password) < 8:
        return False, 'Password must be at least 8 characters long.'
    return True, ''


def normalize_phone(phone_number):
    cleaned = ''.join(ch for ch in str(phone_number or '').strip() if ch.isdigit() or ch == '+')
    return cleaned


def validate_email(value):
    normalized = (value or '').strip().lower()
    return normalized and ('@' in normalized) and ('.' in normalized)


def create_user(username, email, password, phone_number=None, age=None, gender=None, section=None):
    normalized_username = username.strip()
    normalized_email = email.strip().lower()
    normalized_phone = normalize_phone(phone_number)
    normalized_gender = (gender or '').strip()
    normalized_section = (section or '').strip()

    if len(normalized_username) < 3:
        return False, 'Username must be at least 3 characters long.'
    if not validate_email(normalized_email):
        return False, 'Please enter a valid email address.'
    password_ok, password_message = validate_password_strength(password)
    if not password_ok:
        return False, password_message

    age_value = None
    if str(age or '').strip():
        try:
            age_value = int(age)
        except ValueError:
            return False, 'Age must be a number.'
        if age_value < 1 or age_value > 120:
            return False, 'Age must be between 1 and 120.'

    if normalized_phone and len(normalized_phone) < 8:
        return False, 'Phone number seems too short.'

    password_hash = hash_password(password)
    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO users (username, email, password_hash, phone_number, age, gender, section)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_username,
                normalized_email,
                password_hash,
                normalized_phone or None,
                age_value,
                normalized_gender or None,
                normalized_section or None,
            ),
        )
        connection.commit()
        return True, 'Registration successful. You can now sign in.'
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if 'username' in message:
            return False, 'Username is already taken. Please choose another one.'
        if 'email' in message:
            return False, 'Email is already registered. Please sign in instead.'
        if 'phone' in message:
            return False, 'Phone number is already associated with another account.'
        return False, 'Could not create account. Please try again.'
    finally:
        connection.close()


def authenticate_user(identifier, password):
    lookup_value = identifier.strip()
    if not lookup_value:
        return None

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT id, username, email, phone_number, age, gender, section, password_hash
            FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
            """,
            (lookup_value, lookup_value),
        )
        user = cursor.fetchone()
    finally:
        connection.close()

    if user and verify_password(password, user['password_hash']):
        return {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'phone_number': user['phone_number'],
            'age': user['age'],
            'gender': user['gender'],
            'section': user['section'],
        }
    return None


def get_google_oauth_settings():
    client_id = _get_config_value('GOOGLE_CLIENT_ID', 'auth', 'google', 'client_id').strip()
    if not client_id:
        client_id = _get_config_value('GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_ID')

    client_secret = _get_config_value('GOOGLE_CLIENT_SECRET', 'auth', 'google', 'client_secret').strip()
    if not client_secret:
        client_secret = _get_config_value('GOOGLE_CLIENT_SECRET', 'GOOGLE_CLIENT_SECRET')

    redirect_uri = _get_config_value('GOOGLE_REDIRECT_URI', 'auth', 'google', 'redirect_uri').strip()
    if not redirect_uri:
        redirect_uri = _get_config_value('GOOGLE_REDIRECT_URI', 'GOOGLE_REDIRECT_URI')
    if not redirect_uri:
        redirect_uri = 'http://localhost:8501'

    if not client_id or not client_secret or not redirect_uri:
        return None
    return {
        'client_id': client_id,
        'client_secret': client_secret,
        'redirect_uri': redirect_uri,
    }


def build_google_auth_url(state_token):
    settings = get_google_oauth_settings()
    if not settings:
        return None

    params = {
        'client_id': settings['client_id'],
        'redirect_uri': settings['redirect_uri'],
        'response_type': 'code',
        'scope': 'openid email profile',
        'access_type': 'online',
        'include_granted_scopes': 'true',
        'prompt': 'select_account',
        'state': state_token,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def verify_google_code(authorization_code):
    settings = get_google_oauth_settings()
    if not settings:
        return False, 'Google auth is not configured on this server.', None

    try:
        from google.oauth2 import id_token as google_id_token  # pyright: ignore[reportMissingImports]
        from google.auth.transport import requests as google_requests  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False, 'Google auth dependencies are missing. Install requirements and restart the app.', None

    token_payload = {
        'code': authorization_code,
        'client_id': settings['client_id'],
        'client_secret': settings['client_secret'],
        'redirect_uri': settings['redirect_uri'],
        'grant_type': 'authorization_code',
    }

    try:
        token_response = requests.post(GOOGLE_TOKEN_URL, data=token_payload, timeout=12)
        token_response.raise_for_status()
        token_data = token_response.json()
    except Exception as exc:
        return False, f'Google login failed while fetching token: {exc}', None

    raw_id_token = token_data.get('id_token')
    if not raw_id_token:
        return False, 'Google login failed: missing ID token.', None

    try:
        try:
            token_info = google_id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                settings['client_id'],
                clock_skew_in_seconds=60,
            )
        except TypeError:
            token_info = google_id_token.verify_oauth2_token(
                raw_id_token,
                google_requests.Request(),
                settings['client_id'],
            )
    except Exception as exc:
        return False, f'Google login failed: invalid ID token ({exc}). Check that the Google client ID, client secret, and redirect URI match the OAuth app exactly.', None

    if not token_info.get('email_verified'):
        return False, 'Google account email is not verified.', None

    return True, '', {
        'sub': token_info.get('sub'),
        'email': (token_info.get('email') or '').strip().lower(),
        'name': (token_info.get('name') or '').strip(),
    }


def generate_unique_username(base_name):
    normalized = ''.join(ch for ch in (base_name or '').lower() if ch.isalnum() or ch in ('_', '.'))
    normalized = normalized[:24] if normalized else 'google_user'

    connection = get_db_connection()
    try:
        candidate = normalized
        counter = 1
        while True:
            existing = connection.execute(
                "SELECT id FROM users WHERE lower(username) = lower(?) LIMIT 1",
                (candidate,),
            ).fetchone()
            if not existing:
                return candidate
            counter += 1
            candidate = f"{normalized}_{counter}"
    finally:
        connection.close()


def get_or_create_google_user(google_profile):
    google_sub = (google_profile or {}).get('sub')
    email = (google_profile or {}).get('email', '').strip().lower()
    name = (google_profile or {}).get('name', '').strip()

    if not google_sub or not validate_email(email):
        return None

    connection = get_db_connection()
    try:
        user = connection.execute(
            """
            SELECT id, username, email, phone_number, age, gender, section
            FROM users
            WHERE oauth_provider = 'google' AND oauth_sub = ?
            LIMIT 1
            """,
            (google_sub,),
        ).fetchone()
        if user:
            return dict(user)

        user_by_email = connection.execute(
            """
            SELECT id, username, email, phone_number, age, gender, section
            FROM users
            WHERE lower(email) = lower(?)
            LIMIT 1
            """,
            (email,),
        ).fetchone()
        if user_by_email:
            connection.execute(
                """
                UPDATE users
                SET oauth_provider = 'google', oauth_sub = ?
                WHERE id = ?
                """,
                (google_sub, user_by_email['id']),
            )
            connection.commit()
            return dict(user_by_email)

        base_username = (email.split('@')[0] if '@' in email else '') or name or 'google_user'
        username = generate_unique_username(base_username)
        random_password_hash = hash_password(secrets.token_urlsafe(32))

        connection.execute(
            """
            INSERT INTO users (username, email, password_hash, oauth_provider, oauth_sub)
            VALUES (?, ?, ?, 'google', ?)
            """,
            (username, email, random_password_hash, google_sub),
        )
        connection.commit()

        created = connection.execute(
            """
            SELECT id, username, email, phone_number, age, gender, section
            FROM users
            WHERE lower(email) = lower(?)
            LIMIT 1
            """,
            (email,),
        ).fetchone()
        return dict(created) if created else None
    finally:
        connection.close()


def get_user_by_id(user_id):
    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT id, username, email, phone_number, age, gender, section, created_at
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (user_id,),
        )
        user = cursor.fetchone()
        return dict(user) if user else None
    finally:
        connection.close()


def update_user_profile(user_id, username, email, phone_number, age, gender, section):
    normalized_username = (username or '').strip()
    normalized_email = (email or '').strip().lower()
    normalized_phone = normalize_phone(phone_number)
    normalized_gender = (gender or '').strip()
    normalized_section = (section or '').strip()

    if len(normalized_username) < 3:
        return False, 'Username is required and must be at least 3 characters.'
    if not validate_email(normalized_email):
        return False, 'A valid email is required.'
    if not normalized_phone:
        return False, 'Phone number is required.'
    if len(normalized_phone) < 8:
        return False, 'Phone number seems too short.'

    age_value = None
    if str(age or '').strip():
        try:
            age_value = int(age)
        except ValueError:
            return False, 'Age must be a number.'
        if age_value < 1 or age_value > 120:
            return False, 'Age must be between 1 and 120.'

    connection = get_db_connection()
    try:
        connection.execute(
            """
            UPDATE users
            SET username = ?, email = ?, phone_number = ?, age = ?, gender = ?, section = ?
            WHERE id = ?
            """,
            (
                normalized_username,
                normalized_email,
                normalized_phone,
                age_value,
                normalized_gender or None,
                normalized_section or None,
                user_id,
            ),
        )
        connection.commit()
        return True, 'Profile updated successfully.'
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if 'username' in message:
            return False, 'This username is already used by another account.'
        if 'email' in message:
            return False, 'This email is already used by another account.'
        if 'phone' in message:
            return False, 'This phone number is already used by another account.'
        return False, 'Could not update profile. Please try again.'
    finally:
        connection.close()


def create_password_reset_token(user_id):
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.utcnow() + timedelta(minutes=RESET_TOKEN_TTL_MINUTES)).isoformat()

    connection = get_db_connection()
    try:
        connection.execute(
            """
            INSERT INTO password_reset_tokens (user_id, token, expires_at)
            VALUES (?, ?, ?)
            """,
            (user_id, token, expires_at),
        )
        connection.commit()
        return token
    finally:
        connection.close()


def send_password_reset_email(recipient_email, reset_link):
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    mail_from = os.getenv('SMTP_FROM', smtp_user or '')

    if not smtp_host or not smtp_user or not smtp_password or not mail_from:
        return False, 'SMTP is not configured. Generated reset link is shown below for development.'

    message = EmailMessage()
    message['Subject'] = 'AI Skin Detective - Password Reset Link'
    message['From'] = mail_from
    message['To'] = recipient_email
    message.set_content(
        """
You requested a password reset.

Use the link below to reset your password:
{link}

This link expires in {ttl} minutes.
If you did not request this, you can ignore this email.
        """.strip().format(link=reset_link, ttl=RESET_TOKEN_TTL_MINUTES)
    )

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        return True, 'Password reset link has been sent to your registered email.'
    except Exception:
        return False, 'Could not send email at this moment. Generated reset link is shown below for development.'


def request_password_reset(identifier):
    lookup_value = (identifier or '').strip()
    if not lookup_value:
        return False, 'Enter username, email, or phone number.', None

    normalized_phone = normalize_phone(lookup_value)

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT id, username, email
            FROM users
            WHERE lower(username) = lower(?)
               OR lower(email) = lower(?)
               OR phone_number = ?
            LIMIT 1
            """,
            (lookup_value, lookup_value, normalized_phone),
        )
        user = cursor.fetchone()
    finally:
        connection.close()

    if not user:
        return False, 'No account found for that username/email/phone.', None

    token = create_password_reset_token(user['id'])
    base_url = os.getenv('APP_BASE_URL', 'http://localhost:8501')
    reset_link = f"{base_url}?reset_token={token}"
    sent, message = send_password_reset_email(user['email'], reset_link)
    return sent, message, reset_link


def reset_password_with_token(token, new_password):
    password_ok, password_message = validate_password_strength(new_password)
    if not password_ok:
        return False, password_message

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT id, user_id, expires_at, used
            FROM password_reset_tokens
            WHERE token = ?
            LIMIT 1
            """,
            (token,),
        )
        token_row = cursor.fetchone()
        if not token_row:
            return False, 'Invalid reset token.'
        if token_row['used']:
            return False, 'This reset link has already been used.'
        if datetime.utcnow() > datetime.fromisoformat(token_row['expires_at']):
            return False, 'This reset link has expired.'

        new_hash = hash_password(new_password)
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, token_row['user_id']),
        )
        connection.execute(
            "UPDATE password_reset_tokens SET used = 1 WHERE id = ?",
            (token_row['id'],),
        )
        connection.commit()
        return True, 'Password reset successful. You can now sign in.'
    finally:
        connection.close()


def change_user_password(user_id, current_password, new_password):
    password_ok, password_message = validate_password_strength(new_password)
    if not password_ok:
        return False, password_message

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT password_hash
            FROM users
            WHERE id = ?
            LIMIT 1
            """,
            (user_id,),
        )
        user = cursor.fetchone()
        if not user:
            return False, 'User account not found.'

        if not verify_password(current_password, user['password_hash']):
            return False, 'Current password is incorrect.'

        new_hash = hash_password(new_password)
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (new_hash, user_id),
        )
        connection.commit()
        return True, 'Password changed successfully.'
    finally:
        connection.close()


def reset_password_with_identity(identifier, email, new_password):
    lookup_value = identifier.strip()
    lookup_email = email.strip().lower()

    if not lookup_value:
        return False, 'Enter your username or email.'
    if '@' not in lookup_email or '.' not in lookup_email:
        return False, 'Enter a valid registered email address.'

    password_ok, password_message = validate_password_strength(new_password)
    if not password_ok:
        return False, password_message

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            """
            SELECT id
            FROM users
            WHERE (lower(username) = lower(?) OR lower(email) = lower(?))
              AND lower(email) = lower(?)
            LIMIT 1
            """,
            (lookup_value, lookup_value, lookup_email),
        )
        user = cursor.fetchone()
        if not user:
            return False, 'No matching account found for that username/email and email combination.'

        new_hash = hash_password(new_password)
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE id = ?
            """,
            (new_hash, user['id']),
        )
        connection.commit()
        return True, 'Password reset successful. Please login with your new password.'
    finally:
        connection.close()


def initialize_auth_state():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'user_id' not in st.session_state:
        st.session_state['user_id'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = ''
    if 'email' not in st.session_state:
        st.session_state['email'] = ''
    if 'phone_number' not in st.session_state:
        st.session_state['phone_number'] = ''
    if 'age' not in st.session_state:
        st.session_state['age'] = None
    if 'gender' not in st.session_state:
        st.session_state['gender'] = ''
    if 'section' not in st.session_state:
        st.session_state['section'] = ''
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Home'
    if 'main_navigation' not in st.session_state:
        st.session_state['main_navigation'] = '🏠 Home'
    if 'google_oauth_state' not in st.session_state:
        st.session_state['google_oauth_state'] = ''
    if st.session_state.pop('reset_main_navigation', False):
        st.session_state['main_navigation'] = '🏠 Home'


def update_current_page_from_navigation(navigation_map):
    st.session_state['current_page'] = navigation_map.get(
        st.session_state.get('main_navigation'),
        'Home'
    )


def logout_user():
    st.session_state['authenticated'] = False
    st.session_state['user_id'] = None
    st.session_state['username'] = ''
    st.session_state['email'] = ''
    st.session_state['phone_number'] = ''
    st.session_state['age'] = None
    st.session_state['gender'] = ''
    st.session_state['section'] = ''
    st.session_state['current_page'] = 'Home'
    # Defer widget key reset until next run, before sidebar radio is created.
    st.session_state['reset_main_navigation'] = True


def show_auth_page():
    st.markdown("""
    <div style="max-width: 760px; margin: 1rem auto 2rem auto; background: #ffffff; border: 1px solid #d6e8ff; border-radius: 16px; padding: 2rem; box-shadow: 0 10px 24px rgba(0, 102, 204, 0.12);">
        <div style="margin: 0; text-align: center; color: #0a2540; font-size: 1.3rem; line-height: 1;">🔐</div>
        <p style="margin: 0.6rem 0 0 0; text-align: center; color: #476073;">
            Sign in or create an account to use the AI Skin Disease Detection System.
        </p>
    </div>
    """, unsafe_allow_html=True)

    reset_token = st.query_params.get('reset_token')
    if reset_token:
        st.markdown("### Reset Password")
        with st.form('reset_token_form', clear_on_submit=True):
            rp_new_password = st.text_input('New Password', type='password')
            rp_confirm_password = st.text_input('Confirm New Password', type='password')
            rp_submit = st.form_submit_button('Save New Password', use_container_width=True, type='primary')

        if rp_submit:
            if rp_new_password != rp_confirm_password:
                st.error('Passwords do not match.')
            else:
                success, message = reset_password_with_token(reset_token, rp_new_password)
                if success:
                    st.success(message)
                    st.query_params.clear()
                else:
                    st.error(message)
        return

    oauth_error = st.query_params.get('error')
    oauth_code = st.query_params.get('code')
    oauth_state = st.query_params.get('state')

    if oauth_error:
        st.error(f"Google sign-in failed: {oauth_error}")
        st.query_params.clear()
    elif oauth_code:
        expected_state = st.session_state.get('google_oauth_state', '')
        if expected_state and oauth_state != expected_state:
            st.error('Google sign-in failed due to state mismatch. Please try again.')
            st.query_params.clear()
        else:
            ok, message, google_profile = verify_google_code(oauth_code)
            if not ok:
                st.error(message)
                st.query_params.clear()
            else:
                user = get_or_create_google_user(google_profile)
                if not user:
                    st.error('Google sign-in could not create your account.')
                    st.query_params.clear()
                else:
                    st.session_state['authenticated'] = True
                    st.session_state['user_id'] = user['id']
                    st.session_state['username'] = user['username']
                    st.session_state['email'] = user['email']
                    st.session_state['phone_number'] = user.get('phone_number') or ''
                    st.session_state['age'] = user.get('age')
                    st.session_state['gender'] = user.get('gender') or ''
                    st.session_state['section'] = user.get('section') or ''
                    st.session_state['google_oauth_state'] = ''
                    st.query_params.clear()
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()

    if 'auth_mode' not in st.session_state:
        st.session_state['auth_mode'] = 'login'

    auth_col1, auth_col2 = st.columns(2)
    with auth_col1:
        if st.button(
            "Login",
            use_container_width=True,
            type="primary" if st.session_state['auth_mode'] == 'login' else "secondary",
            key="auth_mode_login_btn"
        ):
            st.session_state['auth_mode'] = 'login'
    with auth_col2:
        if st.button(
            "Create Account",
            use_container_width=True,
            type="primary" if st.session_state['auth_mode'] == 'signup' else "secondary",
            key="auth_mode_signup_btn"
        ):
            st.session_state['auth_mode'] = 'signup'

    st.markdown("")

    if st.session_state['auth_mode'] == 'login':
        st.markdown("### Login")

        google_settings = get_google_oauth_settings()
        if google_settings:
            state_token = secrets.token_urlsafe(20)
            st.session_state['google_oauth_state'] = state_token
            auth_url = build_google_auth_url(state_token)

            if not auth_url:
                st.error('Google sign-in is not configured correctly.')
            else:
                st.markdown(
                    """
                    <style>
                    .google-auth-link {
                        width: 100%;
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        gap: 10px;
                        padding: 10px 14px;
                        border: 1px solid #d9dce1;
                        border-radius: 10px;
                        background: #ffffff;
                        color: #1f1f1f !important;
                        font-size: 14px;
                        font-weight: 500;
                        text-decoration: none !important;
                        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
                        transition: all 0.2s ease;
                        margin: 0.15rem 0 0.6rem 0;
                    }
                    .google-auth-link:hover {
                        background: #f8f9fa;
                        border-color: #c7cdd4;
                    }
                    .google-auth-icon {
                        width: 20px;
                        height: 20px;
                        border-radius: 50%;
                        border: 1px solid #e3e6ea;
                        display: inline-flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 12px;
                        font-weight: 700;
                        color: #4285F4;
                        background: #fff;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"""
                    <a class="google-auth-link" href="{auth_url}" target="_self">
                        <span class="google-auth-icon">G</span>
                        <span>Continue with Google / Gmail</span>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption('Google Sign-In is unavailable. Configure GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI to enable it.')

        st.markdown('---')

        with st.form("login_form", clear_on_submit=False):
            login_identifier = st.text_input("Username or Email")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login", use_container_width=True, type="primary")

        if login_submit:
            user = authenticate_user(login_identifier, login_password)
            if user:
                st.session_state['authenticated'] = True
                st.session_state['user_id'] = user['id']
                st.session_state['username'] = user['username']
                st.session_state['email'] = user['email']
                st.session_state['phone_number'] = user.get('phone_number') or ''
                st.session_state['age'] = user.get('age')
                st.session_state['gender'] = user.get('gender') or ''
                st.session_state['section'] = user.get('section') or ''
                st.success(f"Welcome, {user['username']}!")
                st.rerun()
            else:
                st.error("Invalid username/email or password.")

        with st.expander("Forgot Password?"):
            with st.form("forgot_password_form", clear_on_submit=True):
                fp_identifier = st.text_input("Registered Username / Email / Phone")
                fp_submit = st.form_submit_button("Send Reset Link", use_container_width=True)

            if fp_submit:
                success, message, reset_link = request_password_reset(fp_identifier)
                if success:
                    st.success(message)
                else:
                    st.warning(message)

                # For local development when SMTP isn't configured.
                if reset_link:
                    st.caption('Development reset link:')
                    st.code(reset_link)
    else:
        st.markdown("### Create Account")
        with st.form("signup_form", clear_on_submit=True):
            signup_username = st.text_input("Choose Username")
            signup_email = st.text_input("Email")
            signup_phone = st.text_input("Phone Number (Optional)")
            signup_age = st.text_input("Age (Optional)")
            signup_gender = st.selectbox("Gender (Optional)", ["", "Female", "Male", "Other", "Prefer not to say"])
            signup_section = st.text_input("Section (Optional)")
            signup_password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

        if signup_submit:
            if signup_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_user(
                    signup_username,
                    signup_email,
                    signup_password,
                    phone_number=signup_phone,
                    age=signup_age,
                    gender=signup_gender,
                    section=signup_section,
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

# Configure page
st.set_page_config(
    page_title="AI Skin Detective",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Hospital Theme
st.markdown("""
<style>
    /* Import medical icons font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-1: #f4fafc;
        --bg-2: #e6f4fb;
        --text-main: #0f172a;
        --text-muted: #475569;
        --card-bg: #ffffff;
        --card-border: rgba(14, 116, 144, 0.14);
    }
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App background with subtle clinical gradient */
    .stApp {
        background: linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text-main);
    }

    /* Faint medical pattern for depth without visual noise */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(circle at 14% 22%, rgba(2, 132, 199, 0.06) 0, transparent 24%),
            radial-gradient(circle at 82% 16%, rgba(14, 116, 144, 0.05) 0, transparent 20%),
            radial-gradient(circle at 74% 84%, rgba(20, 184, 166, 0.05) 0, transparent 24%);
    }

    /* Keep all page content above overlay */
    .main .block-container {
        position: relative;
        z-index: 1;
        max-width: 1120px;
    }
    
    /* Header styles */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0066cc 0%, #004d99 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-muted);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Medical card container */
    .medical-card {
        background: var(--card-bg);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 102, 204, 0.1);
        border: 1px solid var(--card-border);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .medical-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 102, 204, 0.2);
    }
    
    /* Prediction result boxes */
    .prediction-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #0066cc;
        margin: 1.5rem 0;
        box-shadow: 0 5px 15px rgba(0, 102, 204, 0.15);
    }
    
    .prediction-result {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
        margin: 0.5rem 0;
        text-transform: capitalize;
    }
    
    .confidence-score {
        font-size: 1.5rem;
        color: #28a745;
        font-weight: 600;
    }
    
    /* Urgency boxes */
    .urgent-box {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #d32f2f;
        margin: 1.5rem 0;
        box-shadow: 0 5px 15px rgba(211, 47, 47, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .sidebar-logo {
        animation: float 3s ease-in-out infinite;
    }
    
    .shine-effect {
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
        background-size: 200% 100%;
        animation: shimmer 3s infinite;
    }
    
    .urgent-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: #d32f2f;
        margin: 0;
    }
    
    /* Hospital card */
    .hospital-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .hospital-card:hover {
        border-color: #0066cc;
        box-shadow: 0 8px 20px rgba(0, 102, 204, 0.15);
        transform: translateX(5px);
    }
    
    .hospital-name {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0066cc;
        margin-bottom: 0.5rem;
    }
    
    .hospital-contact {
        font-size: 1.1rem;
        color: #28a745;
        font-weight: 600;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.15);
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255, 152, 0, 0.15);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #03a9f4;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(3, 169, 244, 0.15);
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        border: 3px dashed #0066cc;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: #f0f8ff;
        border-color: #004d99;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0066cc;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling - soft medical palette */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf4ff 0%, #f4fbff 52%, #e9f8ef 100%);
        box-shadow: 4px 0 18px rgba(9, 72, 134, 0.12);
        border-right: 1px solid rgba(0, 102, 204, 0.12);
    }
    
    [data-testid="stSidebar"] * {
        color: #0a2540 !important;
    }
    
    /* Sidebar selectbox styling */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.85);
        border: 1.5px solid rgba(0, 102, 204, 0.18);
        border-radius: 12px;
        padding: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        background: rgba(255, 255, 255, 1);
        border-color: rgba(40, 167, 69, 0.45);
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.12);
    }
    
    /* ── Navigation Radio Cards — equal-width professional buttons ── */
    [data-testid="stSidebar"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.45rem !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #ffffff !important;
        border: 1.5px solid rgba(0, 102, 204, 0.18) !important;
        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: background 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease, transform 0.22s ease !important;
        width: 100% !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        min-height: 48px !important;
        justify-content: flex-start !important;
        box-shadow: 0 2px 8px rgba(0, 102, 204, 0.08) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #f7fcff !important;
        border-color: rgba(40, 167, 69, 0.42) !important;
        box-shadow: 0 5px 14px rgba(0, 102, 204, 0.14) !important;
        transform: translateY(-1px) !important;
    }

    /* Hide the radio circle */
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child > div:first-child {
        display: none !important;
    }

    /* Active / selected item */
    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        background: #eef8ff !important;
        border: 1.5px solid rgba(0, 102, 204, 0.45) !important;
        border-left: 4px solid #28a745 !important;
        box-shadow: 0 6px 16px rgba(0, 102, 204, 0.14) !important;
    }

    /* Label text */
    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {
        color: #0a2540 !important;
        font-size: 0.97rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
        letter-spacing: 0.2px !important;
    }

    /* Legacy radio fallback */
    [data-testid="stSidebar"] .row-widget.stRadio > div {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 10px;
        padding: 0.4rem;
    }

    [data-testid="stSidebar"] .row-widget.stRadio > div > label {
        background: rgba(255, 255, 255, 0.9);
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.25rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1.5px solid rgba(0, 102, 204, 0.15);
        width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stSidebar"] .row-widget.stRadio > div > label:hover {
        background: #f7fcff;
        border-color: rgba(40, 167, 69, 0.35);
    }

    [data-testid="stSidebar"] .row-widget.stRadio > div > label > div:first-child {
        display: none;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #0a2540 !important;
        text-shadow: none;
    }
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {
        border-color: rgba(0, 102, 204, 0.18);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc 0%, #004d99 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 102, 204, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #004d99 0%, #003366 100%);
        box-shadow: 0 6px 20px rgba(0, 102, 204, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        border: 3px dashed #0066cc;
    }
    
    /* Emergency badge */
    .emergency-badge {
        display: inline-block;
        background: #d32f2f;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    /* Specialist badge */
    .specialist-badge {
        display: inline-block;
        background: #0066cc;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0.5rem 0.5rem 0;
    }
    
    /* Progress indicator */
    .analyzing-spinner {
        text-align: center;
        padding: 2rem;
        font-size: 1.2rem;
        color: #0066cc;
        font-weight: 600;
    }

    /* Improve readability for standard text */
    p, li, label, .stMarkdown {
        color: var(--text-main);
        line-height: 1.5;
    }

    /* ---------------- Professional Dashboard Overrides ---------------- */
    :root {
        --bg-1: #f8fcff;
        --bg-2: #eef7f5;
        --bg-3: #f6fbff;
        --sidebar-1: #dcecf9;
        --sidebar-2: #eaf4fb;
        --primary: #0b5f74;
        --accent: #1f4f8a;
        --text-1: #0b1b2b;
        --text-2: #24384a;
        --surface: #ffffff;
        --border: #d6e5f0;
    }

    .stApp {
        background: linear-gradient(125deg, var(--bg-1) 0%, var(--bg-2) 46%, var(--bg-3) 100%);
        color: var(--text-1);
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        background:
            radial-gradient(circle at 12% 18%, rgba(31, 79, 138, 0.07) 0, transparent 26%),
            radial-gradient(circle at 84% 14%, rgba(11, 95, 116, 0.06) 0, transparent 22%),
            radial-gradient(circle at 76% 82%, rgba(16, 185, 129, 0.05) 0, transparent 24%);
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--sidebar-1) 0%, var(--sidebar-2) 100%);
        border-right: 1px solid rgba(31, 79, 138, 0.16);
        box-shadow: 3px 0 18px rgba(31, 79, 138, 0.08);
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: linear-gradient(135deg, #2788a3, #4a82ba) !important;
        border: 1px solid rgba(17, 45, 78, 0.24) !important;
        border-radius: 10px !important;
        padding: 0.72rem 0.9rem !important;
        box-shadow: 0 6px 14px rgba(17, 45, 78, 0.2) !important;
        transition: 0.2s ease all !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        transform: translateY(-1px) !important;
        background: linear-gradient(135deg, #207e99, #4278af) !important;
        border-color: rgba(17, 45, 78, 0.42) !important;
        box-shadow: 0 10px 20px rgba(17, 45, 78, 0.28) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label[aria-checked="true"] {
        background: linear-gradient(135deg, #207e99, #4278af) !important;
        border-left: 4px solid #9ed2ff !important;
        border-color: rgba(158, 210, 255, 0.55) !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label p,
    [data-testid="stSidebar"] [role="radiogroup"] label > div:last-child {
        color: #ffffff !important;
        font-weight: 650 !important;
    }

    .medical-card,
    .hospital-card,
    .metric-card,
    .upload-section,
    .prediction-box,
    .success-box,
    .warning-box,
    .info-box {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        box-shadow: 0 10px 30px rgba(3, 105, 161, 0.1);
    }

    .prediction-box {
        border-left: 6px solid var(--primary);
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: #ffffff !important;
        border: none;
        border-radius: 10px;
        padding: 0.72rem 1.1rem;
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 8px 16px rgba(17, 45, 78, 0.26);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0a5264, #1a4475);
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(17, 45, 78, 0.32);
    }

    .stButton > button:hover * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 12px;
        border: 2px dashed rgba(15, 118, 110, 0.38);
        padding: 0.9rem;
    }

    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }

    h1, h2, h3, h4 {
        color: var(--text-1) !important;
        letter-spacing: 0.2px;
    }

    .sub-header {
        color: var(--text-2) !important;
    }

    .prediction-result {
        color: var(--primary);
    }

    .confidence-score {
        color: #106b42;
    }

    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: #ffffff !important;
        border: 1px solid #cfe0ee !important;
        border-radius: 10px !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    p, li, label, .stMarkdown, .stCaption {
        color: var(--text-2);
        line-height: 1.55;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 0.8rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load hospital data
@st.cache_data
def load_hospital_data():
    """Load hospital and specialist information"""
    hospital_file = os.path.join(SCRIPT_DIR, 'hospital_data.json')
    with open(hospital_file, 'r') as f:
        return json.load(f)

# Load doctor consultation data
@st.cache_data
def load_doctor_consultation_data():
    """Load the local doctor consultation database from CSV or JSON."""
    doctor_csv_file = os.path.join(SCRIPT_DIR, 'doctor_consultation_data.csv')
    if os.path.exists(doctor_csv_file):
        doctor_df = pd.read_csv(doctor_csv_file)
        doctor_records = doctor_df.to_dict(orient='records')
        for doctor in doctor_records:
            languages_value = str(doctor.get('languages_spoken', ''))
            doctor['languages_spoken'] = [
                language.strip()
                for language in languages_value.split('|')
                if language.strip()
            ]
        return doctor_records

    doctor_json_file = os.path.join(SCRIPT_DIR, 'doctor_consultation_data.json')
    with open(doctor_json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_condition_specialist_info(disease_name):
    """Return specialist guidance for the predicted condition."""
    hospital_data = load_hospital_data()
    return hospital_data['disease_specialists'].get(
        disease_name,
        {
            "specialist": "Dermatologist",
            "urgency": "Medium",
            "description": "Consult a dermatologist for proper diagnosis and treatment guidance."
        }
    )

def get_recommended_doctors(disease_name, preferred_language):
    """Return language-matched doctors relevant to the predicted condition."""
    specialist_info = get_condition_specialist_info(disease_name)
    doctors = load_doctor_consultation_data()

    language_matches = []
    selected_language = preferred_language.strip().lower()
    for doctor in doctors:
        spoken_languages = [language.lower() for language in doctor.get('languages_spoken', [])]
        if selected_language in spoken_languages:
            language_matches.append(doctor)

    specialist_text = specialist_info.get('specialist', 'Dermatologist')
    specialist_keywords = [
        keyword.strip().lower()
        for keyword in specialist_text.replace(',', '/').split('/')
        if keyword.strip()
    ]

    specialty_matches = [
        doctor for doctor in language_matches
        if any(keyword in doctor.get('specialization', '').lower() for keyword in specialist_keywords)
    ]

    if not specialty_matches:
        specialty_matches = [
            doctor for doctor in language_matches
            if 'dermat' in doctor.get('specialization', '').lower()
        ]

    return specialist_info, specialty_matches[:6]

def show_recommended_doctors(predicted_class, preferred_language):
    """Render language-based doctor recommendation cards."""
    specialist_info, doctors = get_recommended_doctors(predicted_class, preferred_language)
    specialist_name = specialist_info.get('specialist', 'Dermatologist')

    st.markdown("""
<div style="text-align:center; margin:0.5rem 0 1rem 0;">
<h2 style="color:#0066cc;">👨‍⚕️ Recommended Doctors</h2>
<p style="color:#555; font-size:0.95rem;">Doctors matched to your preferred language and the predicted skin condition</p>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:#eef8ff; border:1px solid #cfe4ff; border-radius:12px; padding:0.95rem 1.15rem; margin-bottom:1rem;">
<p style="margin:0; color:#0a2540; font-size:0.92rem;"><strong>Recommended specialist:</strong> {specialist_name}</p>
<p style="margin:0.25rem 0 0 0; color:#476073; font-size:0.88rem;"><strong>Selected language:</strong> {preferred_language}</p>
</div>
""", unsafe_allow_html=True)

    if not doctors:
        st.info(f"No doctors were found who match {preferred_language} for this consultation type. Please try another language option.")
        return

    for idx, doctor in enumerate(doctors, 1):
        consultation_type = doctor.get('consultation_type', 'Clinic')
        online_available = 'Available' if 'online' in consultation_type.lower() else 'Not Available'
        languages_spoken = ', '.join(doctor.get('languages_spoken', [])) or 'Not available'

        st.markdown(f"""
<div style="background:#ffffff; border:1.5px solid #d6e8ff; border-radius:14px; padding:1.25rem 1.4rem; margin:0.75rem 0; box-shadow:0 4px 14px rgba(0,102,204,0.08);">
<div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:0.75rem;">
<div style="flex:1; min-width:260px;">
<h4 style="color:#0055aa; margin:0 0 0.45rem 0; font-size:1.05rem;">{idx}. {doctor.get('doctor_name', 'Doctor Name Not Available')}</h4>
<p style="margin:0.28rem 0; color:#333; font-size:0.9rem;">🩺 <strong>Specialization:</strong> {doctor.get('specialization', 'Dermatologist')}</p>
<p style="margin:0.28rem 0; color:#555; font-size:0.9rem;">🌐 <strong>Languages Spoken:</strong> {languages_spoken}</p>
</div>
<div style="min-width:220px; background:#f8fcff; border-radius:10px; padding:0.8rem 1rem; border:1px solid #cce0ff;">
<p style="margin:0 0 0.45rem 0; font-size:0.8rem; color:#0066cc; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Consultation</p>
<p style="margin:0.28rem 0; color:#0066cc; font-size:0.92rem; font-weight:700;">📞 <strong>Phone:</strong> {doctor.get('consultation_phone', 'Not available')}</p>
<p style="margin:0.28rem 0; color:#1a8f4a; font-size:0.9rem; font-weight:600;">💻 <strong>Online Availability:</strong> {online_available}</p>
<p style="margin:0.28rem 0; color:#5b6f7f; font-size:0.86rem;">🏥 <strong>Mode:</strong> {consultation_type}</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Save feedback
def save_feedback(feedback_data):
    """Save user feedback to CSV file"""
    feedback_file = os.path.join(os.path.dirname(SCRIPT_DIR), 'results', 'user_feedback.csv')
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([feedback_data])
    
    # Append to CSV (create if doesn't exist)
    if os.path.exists(feedback_file):
        df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        df.to_csv(feedback_file, mode='w', header=True, index=False)
    
    return True

# Create interactive plotly chart
def create_prediction_chart(top_predictions):
    """Create interactive bar chart for predictions"""
    classes = [p[0] for p in top_predictions]
    confidences = [p[1] for p in top_predictions]
    
    # Create color scale (top prediction in different color)
    colors = ['#1f77b4' if i > 0 else '#2ca02c' for i in range(len(classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            y=classes[::-1],
            x=confidences[::-1],
            orientation='h',
            marker=dict(color=colors[::-1]),
            text=[f'{c:.1f}%' for c in confidences[::-1]],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Top 5 Predictions",
        xaxis_title="Confidence (%)",
        yaxis_title="Skin Condition",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

# Main app
def main():
    init_auth_db()
    initialize_auth_state()
    hidden_navigation_option = "__profile_placeholder__"

    if not st.session_state.get('authenticated', False):
        show_auth_page()
        return

    # Apply deferred navigation before sidebar radio is instantiated.
    if st.session_state.pop("navigate_to_detect", False):
        st.session_state["main_navigation"] = "🔬 Detect Disease"
        st.session_state["current_page"] = "Detect"

    # Enhanced Sidebar with better design
    with st.sidebar:
        # Logo/Header Section with animation
        st.markdown("""
        <div class="shine-effect" style="text-align: center; padding: 1.35rem 0.8rem; background: linear-gradient(145deg, #ffffff 0%, #eef8ff 60%, #edf9f1 100%); border-radius: 12px; margin-bottom: 1.2rem; border: 1px solid rgba(0,102,204,0.15); box-shadow: 0 6px 16px rgba(0,102,204,0.10);">
            <h1 class="sidebar-logo" style="color: #0a2540; font-size: 2.45rem; margin: 0; text-shadow: none;">🏥</h1>
            <h3 style="color: #0a2540; margin: 0.45rem 0 0 0; font-size: 1.12rem; font-weight: 800; letter-spacing: 0.4px;">AI Skin Detective</h3>
            <p style="color: #365b7d; font-size: 0.76rem; margin: 0.3rem 0 0 0; font-style: italic; letter-spacing: 0.2px;">Intelligent Dermatology Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu with card design
        st.markdown("### 🧭 Navigation")
        st.markdown(f"👤 Current Profile: **{st.session_state.get('username', 'User')}**")
        profile_clicked = st.button("👤 Profile", use_container_width=True, type="secondary")
        if profile_clicked:
            st.session_state["current_page"] = "Profile"
            st.session_state["main_navigation"] = hidden_navigation_option
        st.markdown("")
        
        # Define navigation options
        nav_options = [
            {"key": "Home", "icon": "🏠", "title": "Home", "color": "#28a745"},
            {"key": "About", "icon": "📖", "title": "About System", "color": "#0066cc"},
            {"key": "Detect", "icon": "🔬", "title": "Detect Disease", "color": "#0ea5e9"},
            {"key": "Feedback", "icon": "📊", "title": "Feedback Hub", "color": "#10b981"}
        ]
        
        # Use radio buttons for navigation
        radio_options = [hidden_navigation_option] + [f"{nav['icon']} {nav['title']}" for nav in nav_options]
        navigation_page_map = {f"{nav['icon']} {nav['title']}": nav['key'] for nav in nav_options}
        navigation_page_map[hidden_navigation_option] = "Profile"

        if st.session_state.get("current_page") == "Profile":
            st.session_state["main_navigation"] = hidden_navigation_option

        if st.session_state.get("main_navigation") not in radio_options:
            st.session_state["main_navigation"] = radio_options[1]
        
        # Display navigation cards with radio
        selected_option = st.radio(
            "nav_menu",
            radio_options,
            label_visibility="collapsed",
            key="main_navigation",
            on_change=update_current_page_from_navigation,
            kwargs={"navigation_map": navigation_page_map}
        )
        
        # Find selected page
        page = st.session_state.get("current_page", navigation_page_map.get(selected_option, "Home"))

        st.markdown("""
        <style>
        [data-testid="stSidebar"] [role="radiogroup"] label:first-child {
            display: none !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown("")

        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <p style="color: rgba(255,255,255,0.6); font-size: 0.75rem; margin: 0;">
                Version 1.0.0<br>
                © 2026 SkinCare AI<br>
                <em>For screening purposes only</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages based on selection
    if page == "Home":
        st.markdown('<div class="main-header">🔬 AI Skin Detective</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Skin Disease Detection &nbsp;|&nbsp; Expert Recommendations &nbsp;|&nbsp; Healthcare Connections</div>', unsafe_allow_html=True)
        show_home_page()
    elif page == "Detect":
        show_detection_page()
    elif page == "About":
        st.markdown('<div class="main-header">🔬 AI Skin Detective</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Skin Disease Detection &nbsp;|&nbsp; Expert Recommendations &nbsp;|&nbsp; Healthcare Connections</div>', unsafe_allow_html=True)
        show_about_page()
    elif page == "Feedback":
        st.markdown('<div class="main-header">🔬 AI Skin Detective</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">AI-Powered Skin Disease Detection &nbsp;|&nbsp; Expert Recommendations &nbsp;|&nbsp; Healthcare Connections</div>', unsafe_allow_html=True)
        show_feedback_history()
    elif page == "Profile":
        st.markdown('<div class="main-header">🔬 AI Skin Detective</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Account Profile &nbsp;|&nbsp; Edit Details</div>', unsafe_allow_html=True)
        show_profile_page()


def show_profile_page():
    """User profile page with editable primary and optional fields."""
    user = get_user_by_id(st.session_state.get('user_id'))
    if not user:
        st.error('Unable to load profile. Please login again.')
        return

    st.markdown("""
    <div class="medical-card">
        <h3 style="color:#0066cc; margin-top:0;">👤 Profile</h3>
        <p style="color:#4b6376; margin-bottom:0;">Username, email, and phone are primary fields and cannot be left empty.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form('profile_update_form', clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            profile_username = st.text_input('Username *', value=user.get('username') or '')
            profile_email = st.text_input('Email *', value=user.get('email') or '')
            profile_phone = st.text_input('Phone Number *', value=user.get('phone_number') or '')
        with col2:
            profile_age = st.text_input('Age (Optional)', value='' if user.get('age') is None else str(user.get('age')))
            profile_gender = st.selectbox(
                'Gender (Optional)',
                ['', 'Female', 'Male', 'Other', 'Prefer not to say'],
                index=['', 'Female', 'Male', 'Other', 'Prefer not to say'].index((user.get('gender') or '')) if (user.get('gender') or '') in ['', 'Female', 'Male', 'Other', 'Prefer not to say'] else 0,
            )
            profile_section = st.text_input('Section (Optional)', value=user.get('section') or '')

        submitted = st.form_submit_button('Save Profile', type='primary', use_container_width=True)

    if submitted:
        success, message = update_user_profile(
            st.session_state.get('user_id'),
            profile_username,
            profile_email,
            profile_phone,
            profile_age,
            profile_gender,
            profile_section,
        )
        if success:
            refreshed_user = get_user_by_id(st.session_state.get('user_id'))
            st.session_state['username'] = refreshed_user.get('username') or ''
            st.session_state['email'] = refreshed_user.get('email') or ''
            st.session_state['phone_number'] = refreshed_user.get('phone_number') or ''
            st.session_state['age'] = refreshed_user.get('age')
            st.session_state['gender'] = refreshed_user.get('gender') or ''
            st.session_state['section'] = refreshed_user.get('section') or ''
            st.success(message)
        else:
            st.error(message)

    st.markdown("---")
    st.markdown("### 🔑 Change Password")
    with st.form("profile_page_change_password_form", clear_on_submit=True):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_new_password = st.text_input("Confirm New Password", type="password")
        change_password_submit = st.form_submit_button("Update Password", use_container_width=True)

    if change_password_submit:
        if new_password != confirm_new_password:
            st.error("New password and confirmation do not match.")
        else:
            success, message = change_user_password(
                st.session_state.get('user_id'),
                current_password,
                new_password,
            )
            if success:
                st.success(message)
            else:
                st.error(message)

    if st.button("🚪 Logout", use_container_width=True, key="profile_logout_button"):
        logout_user()
        st.rerun()

def show_home_page():
    """Home page with professional medical dashboard layout"""

    class_names = load_class_names()

    # 1) Hero Section (Title + short introduction + CTA)
    st.markdown("""
<div style="background: linear-gradient(145deg, #cfe4fa 0%, #cff0e4 58%, #deeffc 100%); border-radius: 18px; padding: 2.6rem 2rem 2rem 2rem; border: 1px solid rgba(31, 79, 138, 0.26); box-shadow: 0 10px 28px rgba(31, 79, 138, 0.16); text-align: center;">
<div style="display: inline-block; background: #d6e8fc; color: #174780; border: 1px solid #9fc3e8; border-radius: 999px; padding: 0.3rem 1rem; font-size: 0.78rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 0.9rem;">AI DERMATOLOGY PLATFORM</div>
<h1 style="margin: 0; color: #0a2540; font-size: 2.7rem; font-weight: 800;">🔬 AI Skin Detective</h1>
<p style="margin: 0.7rem 0 0 0; color: #2c5b7a; font-size: 1rem; font-weight: 600;">AI-Powered Skin Disease Detection | Expert Recommendations | Healthcare Connections</p>
<p style="max-width: 760px; margin: 1rem auto 0 auto; color: #4b6376; line-height: 1.75; font-size: 1rem;">
AI Skin Detective provides fast dermatological screening support through deep learning-based image analysis, confidence scoring, and specialist-oriented healthcare guidance to help users make timely and informed decisions.
</p>
</div>
""", unsafe_allow_html=True)

    cta_left, cta_center, cta_right = st.columns([1, 1.3, 1])
    with cta_center:
        if st.button("🔬 Start Diagnosis", type="primary", use_container_width=True):
            st.session_state["navigate_to_detect"] = True
            st.rerun()

    st.divider()

    # 2) System Highlights / Key Stats
    st.markdown("<h3 style='text-align:center; color:#0a2540;'>💡 How This App Helps You</h3>", unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    highlights = [
        (
            "🔬",
            "Check Skin Conditions Quickly",
            "Upload a photo and get an AI-based screening result in seconds.",
            "linear-gradient(145deg, #d8eafb 0%, #d8efe5 58%, #e6f2fc 100%)",
            "#1b4f87",
        ),
        (
            "📊",
            "Understand Possible Conditions",
            "See possible skin conditions along with confidence levels.",
            "linear-gradient(145deg, #d8eafb 0%, #d8efe5 58%, #e6f2fc 100%)",
            "#1b4f87",
        ),
        (
            "🏥",
            "Find Medical Help Faster",
            "Get suggestions for dermatologists or hospitals if needed.",
            "linear-gradient(145deg, #d8eafb 0%, #d8efe5 58%, #e6f2fc 100%)",
            "#1b4f87",
        ),
        (
            "🧠",
            "Learn About Skin Health",
            "Receive helpful tips and guidance based on the analysis.",
            "linear-gradient(145deg, #d8eafb 0%, #d8efe5 58%, #e6f2fc 100%)",
            "#1b4f87",
        ),
    ]
    for col, (icon, title, desc, bg, accent) in zip([h1, h2, h3, h4], highlights):
        with col:
            st.markdown(f"""
<div style="background:{bg}; border:1px solid #b8d5ee; border-radius:14px; padding:1.1rem; text-align:center; height:198px; box-sizing:border-box; box-shadow: 0 4px 12px rgba(31,79,138,0.12); display:flex; flex-direction:column; align-items:center;">
<div style="font-size:1.45rem; margin-bottom:0.35rem; line-height:1;">{icon}</div>
<div style="font-size:0.95rem; font-weight:800; color:{accent}; line-height:1.35; min-height:52px; display:flex; align-items:center; justify-content:center;">{title}</div>
<div style="font-size:0.82rem; color:#496175; margin-top:0.35rem; line-height:1.45; min-height:64px; display:flex; align-items:flex-start; justify-content:center;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # 3) Core Features
    st.markdown("<h3 style='text-align:center; color:#0a2540;'>🧩 Core Features</h3>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    features = [
        ("🔬", "Smart Diagnosis", "Instant AI-powered image analysis for rapid screening support."),
        ("🏥", "Hospital Matching", "Find the right specialists and healthcare facilities quickly."),
        ("📈", "Confidence Scoring", "Transparent class probability insights for each prediction."),
        ("💡", "Health Guidance", "Personalized recommendations and preventive care suggestions."),
    ]
    for col, (icon, title, desc) in zip([f1, f2, f3, f4], features):
        with col:
            st.markdown(f"""
<div style="background:linear-gradient(145deg, #d8eafb 0%, #d8efe5 58%, #e6f2fc 100%); border:1px solid #b8d5ee; border-radius:14px; padding:1.2rem 1rem; min-height:158px; text-align:center; box-shadow:0 4px 12px rgba(31,79,138,0.12);">
<div style="font-size:1.65rem;">{icon}</div>
<div style="margin-top:0.35rem; color:#0a2540; font-weight:700; font-size:0.95rem;">{title}</div>
<div style="margin-top:0.4rem; color:#3f5668; font-size:0.8rem; line-height:1.45;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # 4) About the Platform
    st.markdown("<h3 style='text-align:center; color:#0a2540;'>🩺 About the Platform</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:white; border-radius:16px; padding:1.5rem 1.6rem; border:1px solid #d7ebff; box-shadow:0 5px 15px rgba(0,102,204,0.08);">
<p style="line-height:1.85; color:#435b6d; font-size:0.95rem; margin:0;">
<strong>AI Skin Detective</strong> is a clinical-support platform designed to make early dermatological screening accessible to everyone. By combining advanced deep learning with a curated medical database, it provides meaningful insights that empower patients and support healthcare professionals in making informed decisions.
</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # 5) Detectable Conditions
    st.markdown("<h3 style='text-align:center; color:#0a2540;'>🔬 Detectable Conditions</h3>", unsafe_allow_html=True)

    # Keep this content unchanged as requested.
    st.markdown(f"""
<div class="prediction-box" style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);">
    <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">
        {len(class_names)} Skin Conditions Supported
    </h4>
    <p style="margin: 0; color: #666; font-size: 0.9rem;">
        Our AI model is trained to identify a comprehensive range of dermatological conditions
    </p>
</div>
""", unsafe_allow_html=True)

    condition_categories = {
        "Inflammatory Conditions": [
            "Acne",
            "Eczema",
            "Psoriasis",
            "Rosacea",
            "Lichen",
            "Lupus",
        ],
        "Infections & Infestations": [
            "Candidiasis",
            "Tinea",
            "Infestations & Bites",
        ],
        "Tumors & Growths": [
            "Benign Tumors",
            "Vascular Tumors",
            "Moles",
            "Seborrheic Keratoses",
            "Skin Cancer",
        ],
    }

    cc1, cc2, cc3 = st.columns(3)
    category_cols = [cc1, cc2, cc3]
    for col, (category, items) in zip(category_cols, condition_categories.items()):
        with col:
            item_html = "".join(
                [
                    f"<div style=\"background:white; padding:0.45rem 0.68rem; margin:0.24rem 0; border-radius:8px; border-left:3px solid #2f83d5; font-size:0.82rem; color:#334c5e; box-shadow:0 1px 3px rgba(0,0,0,0.04);\">• {name}</div>"
                    for name in items
                ]
            )
            st.markdown(f"""
<div style="background:#f8fcff; border:1px solid #dceefe; border-radius:12px; padding:0.85rem; min-height:280px; box-shadow:0 2px 8px rgba(0,102,204,0.05);">
    <div style="text-align:center; font-size:0.86rem; font-weight:700; color:#0066cc; margin-bottom:0.6rem;">{category}</div>
    {item_html}
</div>
""", unsafe_allow_html=True)

    st.divider()

    # 6) Medical Disclaimer
    st.markdown("<h3 style='text-align:center; color:#0a2540;'>⚠️ Medical Disclaimer</h3>", unsafe_allow_html=True)
    st.markdown("""
<div class="warning-box" style="
    background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
    border-left: 5px solid #ff9800;
    padding: 1.5rem;
    border-radius: 10px;
">
    <h3 style="color: #ff6f00; margin-top: 0;">⚠️ Important Medical Disclaimer</h3>
    <p style="color: #555; line-height: 1.8;">
        <strong>This is an AI-based screening tool designed to assist with early detection.
        It is NOT a substitute for professional medical diagnosis.</strong>
    </p>
    <ul style="color: #555; line-height: 1.8;">
        <li>Always consult a qualified healthcare professional for accurate diagnosis</li>
        <li>Use this tool for preliminary screening and awareness only</li>
        <li>Seek immediate medical attention for serious or worsening symptoms</li>
        <li>AI predictions are probabilistic estimates, not definitive diagnoses</li>
        <li>Do not delay professional medical care based on these results</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # 7) How to Use the System (content unchanged)
    st.markdown("""
<div class="medical-card" style="margin-top: 1rem;">
    <h3 style="color: #0066cc; margin-top: 0;">🚀 How to Use the System</h3>
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
        <div style="margin: 1rem 0; padding-left: 1rem; border-left: 4px solid #0066cc;">
            <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">1. Upload Image</h4>
            <p style="margin: 0; color: #666;">Navigate to "Detect Disease" and upload a clear photo</p>
        </div>
        <div style="margin: 1rem 0; padding-left: 1rem; border-left: 4px solid #0066cc;">
            <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">2. AI Analysis</h4>
            <p style="margin: 0; color: #666;">Our system analyzes the image using deep learning</p>
        </div>
        <div style="margin: 1rem 0; padding-left: 1rem; border-left: 4px solid #0066cc;">
            <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">3. Review Results</h4>
            <p style="margin: 0; color: #666;">Get predictions, confidence scores, and recommendations</p>
        </div>
        <div style="margin: 1rem 0; padding-left: 1rem; border-left: 4px solid #0066cc;">
            <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">4. Seek Professional Care</h4>
            <p style="margin: 0; color: #666;">Contact recommended healthcare facilities</p>
        </div>
        <div style="margin: 1rem 0; padding-left: 1rem; border-left: 4px solid #0066cc;">
            <h4 style="color: #0066cc; margin: 0 0 0.5rem 0;">5. Provide Feedback</h4>
            <p style="margin: 0; color: #666;">Help us improve by sharing your experience</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics (if feedback exists)
    feedback_file = os.path.join(os.path.dirname(SCRIPT_DIR), 'results', 'user_feedback.csv')
    if os.path.exists(feedback_file):
        try:
            df = pd.read_csv(feedback_file)
            if len(df) > 0:
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0 1rem 0;">
                    <h2 style="color: #0066cc;">📊 System Performance Metrics</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(df)}</div>
                        <div class="metric-label">Total Analyses</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    avg_accuracy = df['accuracy_rating'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_accuracy:.1f}/5</div>
                        <div class="metric-label">Diagnostic Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    avg_satisfaction = df['satisfaction_rating'].mean()
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{avg_satisfaction:.1f}/5</div>
                        <div class="metric-label">User Satisfaction</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    would_recommend = (df['would_recommend'] == 'Yes').sum() / len(df) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{would_recommend:.0f}%</div>
                        <div class="metric-label">Recommendation Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
        except:
            pass

def show_detection_page():
    """Main detection page with enhanced hospital theme"""
    
    # Page header with medical theme
    st.markdown('<div class="main-header">🏥 Medical Imaging Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Dermatological Assessment System</div>', unsafe_allow_html=True)

    # Instructions section
    st.markdown("""
    <div class="medical-card">
        <h3 style="color: #0066cc; margin-top: 0;">📋 Before You Begin</h3>
        <ul style="line-height: 2;">
            <li>✓ Use a clear, well-lit photograph of the affected skin area</li>
            <li>✓ Ensure the image is in focus and shows the condition clearly</li>
            <li>✓ Supported formats: JPG, JPEG, PNG</li>
            <li>✓ Recommended size: At least 500x500 pixels</li>
        </ul>
        <div class="warning-box">
            <strong>⚠️ Medical Disclaimer:</strong> This is an AI screening tool for educational purposes only. 
            Always consult a qualified healthcare professional for accurate diagnosis and treatment.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced file uploader section
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #0066cc;">📸 Upload Medical Image</h2>
        <p style="color: #666; font-size: 1.1rem;">Drag and drop or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload skin image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the skin condition",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Create two-column layout for image and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: #0066cc; margin-top: 0;">📷 Uploaded Medical Image</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display uploaded image in a styled container
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Patient Skin Image")
        
        with col2:
            st.markdown("""
            <div class="medical-card">
                <h3 style="color: #0066cc; margin-top: 0;">📊 Image Details</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display image information
            st.markdown(f"""
            <div class="info-box">
                <p><strong>File Name:</strong><br>{uploaded_file.name}</p>
                <p><strong>Dimensions:</strong><br>{image.size[0]} × {image.size[1]} pixels</p>
                <p><strong>Format:</strong><br>{image.format}</p>
                <p style="margin-bottom: 0;"><strong>Status:</strong><br>
                    <span style="color: #28a745; font-weight: 600;">✓ Ready for Analysis</span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prominent analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_clicked = st.button(
                "🔬 START MEDICAL ANALYSIS",
                type="primary",
                use_container_width=True
            )
        
        if analyze_clicked:
            # Show analysis progress with medical theme
            with st.spinner(""):
                st.markdown("""
                <div class="analyzing-spinner">
                    <div style="font-size: 3rem;">🔬</div>
                    <div style="margin-top: 1rem;">Analyzing Medical Image...</div>
                    <div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
                        Processing through AI diagnostic system
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Save uploaded file temporarily
                temp_dir = os.path.join(SCRIPT_DIR, 'temp')
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_file, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Make prediction
                try:
                    result = predict_skin_disease(temp_file)
                    
                    # Store in session state
                    st.session_state['prediction_result'] = result
                    st.session_state['analyzed'] = True
                    st.session_state['image_name'] = uploaded_file.name
                    
                    # Clean up temp file
                    os.remove(temp_file)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-box">
                        <h3 style="color: #28a745; margin: 0;">✅ Analysis Complete!</h3>
                        <p style="margin: 0.5rem 0 0 0;">Medical imaging analysis has been successfully completed.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="urgent-box">
                        <h3 style="color: #d32f2f; margin: 0;">❌ Analysis Error</h3>
                        <p style="margin: 0.5rem 0 0 0;">Unable to process image: {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        
        # Display results if analysis is complete
        if st.session_state.get('analyzed', False):
            st.divider()
            display_results()

def display_results():
    """Display prediction results with enhanced hospital theme"""
    result = st.session_state.get('prediction_result')
    if result is None:
        return
    
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Diagnostic Header
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h1 style="color: #0066cc; font-size: 2.5rem; margin: 0;">
            🏥 Medical Diagnostic Report
        </h1>
        <p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">
            AI-Powered Dermatological Analysis Results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Results Section
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Primary Diagnosis Box
        if confidence > 70:
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #0066cc; margin-top: 0; font-size: 1.5rem;">
                    🔬 Primary Diagnosis
                </h2>
                <div class="prediction-result">{predicted_class.replace('_', ' ')}</div>
                <div class="confidence-score">
                    Confidence Level: {confidence:.1f}%
                </div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 2px solid #0066cc66;">
                    <div style="background: #0066cc; height: 8px; border-radius: 4px; width: {confidence}%;"></div>
                    <p style="color: #666; margin-top: 0.5rem; font-size: 0.9rem;">
                        Diagnostic Certainty: <strong>{'Very High' if confidence > 90 else 'High' if confidence > 75 else 'Moderate'}</strong>
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h2 style="color: #ff9800; margin-top: 0; font-size: 1.5rem;">
                    ⚠️ Preliminary Assessment
                </h2>
                <div style="font-size: 1.5rem; font-weight: 700; color: #ff9800; margin: 0.5rem 0;">
                    {predicted_class.replace('_', ' ')}
                </div>
                <div style="font-size: 1.2rem; color: #666; font-weight: 600;">
                    Confidence: {confidence:.1f}%
                </div>
                <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px;">
                    <p style="margin: 0; color: #666;">
                        ⚠️ <strong>Low Confidence Detected</strong><br>
                        Multiple conditions possible. Professional consultation strongly recommended.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Alternative Diagnoses
        st.markdown("""
        <div class="medical-card">
            <h3 style="color: #0066cc; margin-top: 0;">📋 Alternative Differential Diagnoses</h3>
            <p style="color: #666; margin-bottom: 1rem;">Other possible conditions to consider:</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, (cls, conf) in enumerate(result['top_5_predictions'][1:], 2):
            confidence_color = "#28a745" if conf > 20 else "#ffc107" if conf > 10 else "#999"
            st.markdown(f"""
            <div class="hospital-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #666; font-weight: 600;">#{i}</span>
                        <span style="color: #333; font-size: 1.1rem; margin-left: 0.5rem;">
                            {cls.replace('_', ' ')}
                        </span>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {confidence_color}; font-weight: 700; font-size: 1.2rem;">
                            {conf:.1f}%
                        </div>
                        <div style="background: {confidence_color}; height: 4px; border-radius: 2px; width: {conf*2}px; margin-top: 0.3rem;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Interactive chart with hospital theme
        st.markdown("""
        <div class="medical-card">
            <h3 style="color: #0066cc; margin-top: 0;">📊 Diagnostic Probability Chart</h3>
        </div>
        """, unsafe_allow_html=True)
        
        fig = create_prediction_chart(result['top_5_predictions'])
        
        # Customize the chart for hospital theme
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12, color="#333"),
            title_font=dict(size=16, color="#0066cc", family="Inter, sans-serif"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Skin Health Advice
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h2 style="color: #0066cc;">💡 Skin Health Advice</h2>
    </div>
    """, unsafe_allow_html=True)
    
    recommendation = get_health_recommendation(predicted_class)
    st.markdown(f"""
    <div class="info-box">
        <div style="display: flex; align-items: start;">
            <div style="font-size: 2rem; margin-right: 1rem;">💡</div>
            <div>
                <h4 style="color: #03a9f4; margin: 0 0 0.5rem 0;">Treatment Guidance</h4>
                <p style="margin: 0; line-height: 1.8;">{recommendation}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    hospital_data = {
        "Thiruvananthapuram": [
            "Government Medical College Hospital Thiruvananthapuram",
            "KIMSHEALTH Hospital",
            "NIMS Medicity",
            "Dr Somervell Memorial CSI Medical College Hospital",
            "Sree Gokulam Medical College Hospital"
        ],
        "Kollam": [
            "A A Rahim Memorial District Hospital Kollam",
            "Travancore Medical College Hospital",
            "Upasana Hospital",
            "NS Memorial Institute of Medical Sciences"
        ],
        "Pathanamthitta": [
            "MGM Muthoot Medical Centre Kozhencherry",
            "Believers Church Medical College Hospital",
            "Pushpagiri Medical College Hospital"
        ],
        "Alappuzha": [
            "Government Medical College Hospital Alappuzha",
            "TD Medical College Hospital",
            "Sahrudaya Hospital"
        ],
        "Kottayam": [
            "Government Medical College Hospital Kottayam",
            "Caritas Hospital",
            "Believers Church Medical College Hospital",
            "Pushpagiri Medical College Hospital Thiruvalla"
        ],
        "Idukki": [
            "District Hospital Idukki",
            "St Johns Hospital Kattappana",
            "Al Azhar Medical College Hospital"
        ],
        "Ernakulam": [
            "Amrita Institute of Medical Sciences",
            "Aster Medcity",
            "Lakeshore Hospital",
            "Rajagiri Hospital",
            "Government Medical College Ernakulam",
            "Lisie Hospital",
            "Lourdes Hospital"
        ],
        "Thrissur": [
            "Government Medical College Hospital Thrissur",
            "Jubilee Mission Medical College Hospital",
            "Amala Institute of Medical Sciences",
            "Westfort Hospital",
            "Elite Mission Hospital"
        ],
        "Palakkad": [
            "Government Medical College Hospital Palakkad",
            "Palakkad District Hospital",
            "Vasan Eye Care Hospital Palakkad",
            "Al Shifa Hospital Perinthalmanna"
        ],
        "Malappuram": [
            "MES Medical College Hospital",
            "KIMS Al Shifa Hospital",
            "Moulana Hospital Perinthalmanna",
            "PK Das Institute of Medical Sciences"
        ],
        "Kozhikode": [
            "Government Medical College Hospital Kozhikode",
            "Baby Memorial Hospital",
            "KMCT Medical College Hospital",
            "Malabar Medical College Hospital"
        ],
        "Wayanad": [
            "District Hospital Mananthavady",
            "DM WIMS Medical College Hospital",
            "Leo Hospital Kalpetta"
        ],
        "Kannur": [
            "Government Medical College Hospital Kannur",
            "Aster MIMS Hospital Kannur",
            "Indira Gandhi Co Operative Hospital"
        ],
        "Kasaragod": [
            "Government General Hospital Kasaragod",
            "Kanhangad District Hospital",
            "Malik Deenar Charitable Hospital"
        ]
    }

    st.markdown("""
<div style="background:linear-gradient(135deg,#f6fbff 0%,#eef9f3 100%); border-radius:14px; padding:1.2rem 1.4rem 1rem 1.4rem; border:1px solid #d7e9ff; margin-bottom:1rem; box-shadow:0 4px 12px rgba(0,102,204,0.06);">
<h4 style="color:#0a2540; margin:0 0 0.3rem 0;">🏥 Hospitals by District</h4>
<p style="color:#4b6376; font-size:0.88rem; margin:0;">Select a district to view available hospitals.</p>
</div>
""", unsafe_allow_html=True)

    selected_district = st.selectbox(
        "📍 Select District",
        options=sorted(hospital_data.keys()),
        key="selected_hospital_district"
    )

    if selected_district:
        hospital_list = hospital_data.get(selected_district, [])
        if hospital_list:
            hospital_html = "".join(
                [
                    f"<li style='margin:0.35rem 0; color:#334c5e;'>{hospital_name}</li>"
                    for hospital_name in hospital_list
                ]
            )
            st.markdown(f"""
<div style="background:#ffffff; border:1px solid #d6e8ff; border-radius:12px; padding:1rem 1.2rem; margin:0.6rem 0 1rem 0; box-shadow:0 3px 10px rgba(0,102,204,0.06);">
<p style="margin:0 0 0.5rem 0; color:#0a2540; font-weight:700;">Available hospitals in {selected_district}:</p>
<ul style="margin:0; padding-left:1.1rem;">{hospital_html}</ul>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No hospital data available for the selected district.")

    st.markdown("""
<div style="background:linear-gradient(135deg,#eaf4ff 0%,#e6f7ee 100%); border-radius:14px; padding:1.2rem 1.4rem 1rem 1.4rem; border:1px solid #cde3ff; margin-bottom:1rem; box-shadow:0 4px 12px rgba(0,102,204,0.07);">
<h4 style="color:#0a2540; margin:0 0 0.3rem 0;">🗣️ Choose Language for Talking to the Doctor</h4>
<p style="color:#4b6376; font-size:0.88rem; margin:0;">Select the language you prefer to use when speaking with the doctor. We will show doctors who can communicate in that language.</p>
</div>
""", unsafe_allow_html=True)

    language_options = [
        "-- Select Preferred Language --",
        "English",
        "Malayalam",
        "Hindi",
        "Tamil",
        "Kannada",
        "Telugu",
        "Bengali",
        "Marathi",
        "Urdu",
        "Punjabi"
    ]

    preferred_language_choice = st.selectbox(
        "🩺 Language for Doctor Consultation",
        language_options,
        index=0 if st.session_state.get("preferred_language") not in language_options else language_options.index(st.session_state.get("preferred_language")),
        key="preferred_language_select"
    )

    if preferred_language_choice != "-- Select Preferred Language --":
        st.session_state["preferred_language"] = preferred_language_choice
        st.markdown(f"""
<div style="background:#e8f5e9; border-left:4px solid #28a745; border-radius:8px; padding:0.6rem 1rem; margin:0.75rem 0 1rem 0;">
<span style="color:#1a6e30; font-size:0.88rem; font-weight:600;">✅ Doctor consultation language selected: {preferred_language_choice}</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.session_state.pop("preferred_language", None)

    preferred_language = st.session_state.get("preferred_language", "")

    if preferred_language:
        show_recommended_doctors(predicted_class, preferred_language)
    else:
        st.markdown("""
<div style="background:#f0f8ff; border:1px solid #cde3ff; border-radius:12px; padding:1.1rem 1.4rem; margin:1rem 0;">
<h4 style="color:#0066cc; margin:0 0 0.4rem 0;">👨‍⚕️ Recommended Doctors</h4>
<p style="color:#555; margin:0; font-size:0.92rem;">
Please choose your <strong>preferred language for talking to the doctor</strong> here to see recommended doctors after diagnosis.
</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # Feedback Section
    st.header("📝 Your Feedback")
    show_feedback_form(predicted_class, confidence)

def show_feedback_form(predicted_class, confidence):
    """Display enhanced feedback form with hospital theme"""
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1rem 0;">
        <h2 style="color: #0066cc;">📝 Patient Feedback & Follow-up</h2>
        <p style="color: #666;">Your feedback helps us improve diagnostic accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="medical-card">
        <p style="color: #666; margin: 0;">
            Please share your experience and help us enhance the AI diagnostic system. 
            Your input is valuable for improving healthcare outcomes.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("feedback_form"):
        st.markdown("### 📊 Diagnostic Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**How accurate was the AI diagnosis?**")
            accuracy_rating = st.slider(
                "",
                1, 5, 3,
                help="1 = Not accurate, 5 = Very accurate",
                key="accuracy",
                label_visibility="collapsed"
            )
            st.caption(f"{'⭐' * accuracy_rating} ({accuracy_rating}/5)")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("**Overall system satisfaction**")
            satisfaction_rating = st.slider(
                "",
                1, 5, 4,
                help="1 = Not satisfied, 5 = Very satisfied",
                key="satisfaction",
                label_visibility="collapsed"
            )
            st.caption(f"{'⭐' * satisfaction_rating} ({satisfaction_rating}/5)")
        
        with col2:
            st.markdown("**Would you recommend this system?**")
            would_recommend = st.radio(
                "",
                ["Yes", "No", "Maybe"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("**Have you consulted a medical professional?**")
            consulted_doctor = st.radio(
                "",
                ["Not yet", "Yes - Confirmed", "Scheduled Appointment"],
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        st.markdown("### 🏥 Medical Follow-up")
        
        user_diagnosis = st.text_input(
            "Professional Diagnosis (if available)",
            placeholder="Enter the actual diagnosis from your doctor",
            help="This helps us validate and improve AI accuracy"
        )
        
        comments = st.text_area(
            "Additional Comments or Suggestions",
            placeholder="Share your experience, concerns, or suggestions for improvement...",
            height=100
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "📤 SUBMIT FEEDBACK",
                type="primary",
                use_container_width=True
            )
        
        if submitted:
            feedback_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_name': st.session_state.get('image_name', 'unknown'),
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'accuracy_rating': accuracy_rating,
                'satisfaction_rating': satisfaction_rating,
                'would_recommend': would_recommend,
                'consulted_doctor': consulted_doctor,
                'actual_diagnosis': user_diagnosis if user_diagnosis else 'N/A',
                'comments': comments if comments else 'N/A'
            }

            if save_feedback(feedback_data):
                st.session_state['feedback_submit_success'] = True
                st.session_state['feedback_show_balloons'] = True

    if st.session_state.get('feedback_submit_success', False):
        st.markdown("""
        <div class="success-box" style="
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            border-left: 5px solid #28a745;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        ">
            <h4 style="color: #28a745; margin: 0 0 0.5rem 0;">
                ✅ Feedback Submitted Successfully
            </h4>
            <p style="margin: 0; color: #555;">
                Thank you for your valuable feedback! Your input helps us improve
                diagnostic accuracy and provide better healthcare recommendations.
                Your contribution makes a difference in advancing AI-powered healthcare.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.pop('feedback_show_balloons', False):
            st.balloons()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 ANALYZE ANOTHER IMAGE", type="primary", use_container_width=True):
                st.session_state['analyzed'] = False
                st.session_state['feedback_submit_success'] = False
                st.session_state['feedback_show_balloons'] = False
                st.rerun()

def show_about_page():
    """About page"""
    st.header("About This System")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **AI-Based Intelligent System for Skin Disease Detection and Healthcare Recommendation**
    
    This system supports early skin-condition screening using artificial intelligence.
    Users can upload an image, review possible conditions with confidence scores,
    and receive guidance on suitable next steps, including language-matched doctor recommendations.
    
    The platform is designed to improve awareness and support timely clinical consultation.
    
    ### 🔬 Technology Stack
    
    - **Deep Learning Framework:** TensorFlow / Keras
    - **Model Architecture:** MobileNetV2 (transfer learning)
    - **Web Framework:** Streamlit
    - **Data Visualization:** Plotly and Matplotlib
    - **Image Processing:** PIL and NumPy
    
    ### 📊 Dataset
    
    - **Total Classes:** 22 skin conditions
    - **Data Source:** Kaggle Skin Disease Dataset
    - **Data Type:** Labeled dermatoscopic/clinical skin images
    - **Data Split:** Separate training, validation, and testing sets
    - **Purpose:** To support model training, tuning, and objective evaluation
    
    ### 🎓 Detected Conditions Include:
    
    """)
    
    conditions = load_class_names()
    cols = st.columns(4)
    for idx, condition in enumerate(conditions):
        with cols[idx % 4]:
            st.write(f"• {condition.replace('_', ' ')}")
    
    st.markdown("""
    ### ⚙️ Model Performance
    
     - **Architecture:** MobileNetV2 with a custom classification head
     - **Optimizer:** Adam optimizer with learning-rate control
     - **Regularization:** Batch normalization and dropout
     - **Training Strategy:** Transfer learning followed by fine-tuning
     - **Output:** Top predicted conditions with confidence scores
    
    ### 🏗️ System Modules
    
    1. **Data Collection & Preprocessing**
         - Dataset preparation and quality checks
         - Image resizing, normalization, and augmentation
         - Structured train/validation/test partitioning
    
    2. **Model Training & Evaluation**
         - Transfer learning model development
         - Hyperparameter tuning and regularization
         - Performance evaluation using validation/testing results
    
    3. **UI Integration & Feedback**
         - Interactive web interface built with Streamlit
         - AI prediction display with confidence breakdown
         - Language-matched doctor recommendation and feedback capture
    
    4. **Testing & Documentation**
         - Functional testing of model and user workflow
         - User guidance and technical documentation for review
    
    ### ⚠️ Disclaimer
    
     **IMPORTANT:** This tool is intended for educational use and preliminary screening support only.
     It does **not** provide a confirmed medical diagnosis.
    
     Always consult a qualified dermatologist or healthcare professional for diagnosis,
     treatment planning, and clinical follow-up.
    
    ### 📞 Support
    
     For medical concerns, contact a qualified healthcare professional.
    
    ---
    
    **Version:** 1.0.0 | **Last Updated:** March 2026
    """)

def show_feedback_history():
    """Display feedback history and analytics"""
    st.header("📝 Feedback History")
    
    feedback_file = os.path.join(os.path.dirname(SCRIPT_DIR), 'results', 'user_feedback.csv')
    
    if not os.path.exists(feedback_file):
        st.info("No feedback data available yet. Submit some predictions and feedback to see analytics here!")
        return
    
    df = pd.read_csv(feedback_file)
    
    if len(df) == 0:
        st.info("No feedback submitted yet.")
        return
    
    # Summary metrics
    st.subheader("📈 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Submissions", len(df))
    with col2:
        st.metric("Avg Accuracy Rating", f"{df['accuracy_rating'].mean():.2f}/5")
    with col3:
        st.metric("Avg Satisfaction", f"{df['satisfaction_rating'].mean():.2f}/5")
    with col4:
        recommend_pct = (df['would_recommend'] == 'Yes').sum() / len(df) * 100
        st.metric("Would Recommend", f"{recommend_pct:.0f}%")
    
    # Recent feedback
    st.subheader("📝 Recent Feedback")
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    
    for idx, row in recent_df.iterrows():
        with st.expander(f"{row['timestamp']} - {row['predicted_class']} ({row['confidence']:.1f}%)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Accuracy Rating:** {'⭐' * int(row['accuracy_rating'])}")
                st.write(f"**Satisfaction:** {'⭐' * int(row['satisfaction_rating'])}")
                st.write(f"**Would Recommend:** {row['would_recommend']}")
            with col2:
                st.write(f"**Consulted Doctor:** {row['consulted_doctor']}")
                st.write(f"**Actual Diagnosis:** {row['actual_diagnosis']}")
            
            if row['comments'] != 'N/A':
                st.write(f"**Comments:** {row['comments']}")
    
# Run the app
if __name__ == "__main__":
    # Initialize session state
    if 'analyzed' not in st.session_state:
        st.session_state['analyzed'] = False
    
    main()
