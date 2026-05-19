"""
oai_platform_core — shared infrastructure utilities for the OAI Agent Development Kit.

Modules
-------
security.token_manager          Redis/redislite-backed API token management
security.saml_token_validation  SAML response parsing and RSA signature verification
networking                      Host IP address discovery (public, local)
db.base                         Shared async database backend base classes
                                (DatabaseBackend, BasePostgresBackend,
                                 BaseSQLiteBackend, PersistentSQLiteBackend)
"""
