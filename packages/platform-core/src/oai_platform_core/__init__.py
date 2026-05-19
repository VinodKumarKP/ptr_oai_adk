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
security.token_utils            extract_bearer_token() — framework-agnostic
                                API-token extraction from HTTP headers
deployers.base                  BaseDeployer ABC (initialize, shutdown,
                                find_available_port, image_exists)
deployers.env_utils             build_deployment_env() — common deployment
                                env-var builder (DB pool, Redis, port, etc.)
deployers.infra_cli             generate_infra_compose_main() — shared CLI
                                entry-point for infra compose generation
deployers.docker_compose_base   BaseDockerComposeManager — template-method
                                base for Docker Compose deployers; abstract
                                hooks: _build_infra_compose_dict,
                                _build_service_dict, _get_service_env
deployers.python_package_base   BasePythonPackageDeployer — template-method
                                base for Python-subprocess deployers; abstract
                                hooks: _get_service_env, _get_server_py_path,
                                _get_start_command
"""
