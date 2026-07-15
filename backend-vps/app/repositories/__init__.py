"""Repository layer — the only place that talks to the ORM.

Services call these functions; they never build SQLAlchemy queries themselves. Each
call opens its own short-lived session (see ``database.session_scope``).
"""
