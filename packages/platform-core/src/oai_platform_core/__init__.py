"""
oai_platform_core — shared infrastructure utilities for the OAI Agent Development Kit.

Modules
-------
security.token_manager          Redis/redislite-backed API token management
security.saml_token_validation  SAML response parsing, RSA signature verification,
                                and is_saml_token() detection helper
networking                      Host IP address discovery (public, local)
db.base                         Shared async database backend base classes
                                (DatabaseBackend, BasePostgresBackend,
                                 BaseSQLiteBackend, PersistentSQLiteBackend)
exceptions                      Shared exception hierarchy
                                (OAIBaseException, AuthenticationException)
logging_utils                   Shared logging helper (get_logger)
"""
