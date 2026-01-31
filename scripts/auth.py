"""Simple authentication helper for Streamlit pages."""

import streamlit as st
import hashlib

# User database - in production, use st.secrets or environment variables
USERS = {
    "steve": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "Steve"
    },
    "guest": {
        "password_hash": hashlib.sha256("guest".encode()).hexdigest(),
        "role": "viewer",
        "name": "Guest"
    }
}


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


def check_credentials(username: str, password: str) -> dict | None:
    """Check if credentials are valid. Returns user dict or None."""
    user = USERS.get(username.lower())
    if user and user["password_hash"] == hash_password(password):
        return {"username": username, "role": user["role"], "name": user["name"]}
    return None


def login_form():
    """Display login form and handle authentication."""
    if "user" in st.session_state:
        return True

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            user = check_credentials(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Invalid username or password")

    return False


def logout():
    """Log out the current user."""
    if "user" in st.session_state:
        del st.session_state["user"]
    st.rerun()


def get_current_user() -> dict | None:
    """Get the currently logged in user."""
    return st.session_state.get("user")


def require_auth(allowed_roles: list[str] | None = None):
    """
    Check if user is authenticated and has required role.
    Call at the top of each page.
    """
    user = get_current_user()

    if not user:
        st.switch_page("streamlit_app.py")
        st.stop()

    if allowed_roles and user["role"] not in allowed_roles:
        st.error("You don't have permission to access this page.")
        st.caption(f"Logged in as: {user['name']} ({user['role']})")
        if st.button("Go to Fixtures"):
            st.switch_page("pages/1_📅_Fixtures.py")
        st.stop()

    return user


def show_user_info():
    """Show current user info in sidebar with logout button."""
    user = get_current_user()
    if user:
        with st.sidebar:
            st.divider()
            st.caption(f"Logged in as **{user['name']}**")
            if st.button("Logout", use_container_width=True):
                logout()
