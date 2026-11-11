-- Run once as a database administrator. Replace the password placeholder
-- before execution; do not commit the real password.
CREATE ROLE nirf_readonly
    WITH LOGIN
    PASSWORD '<replace-with-a-strong-password>'
    NOSUPERUSER
    NOCREATEDB
    NOCREATEROLE
    NOREPLICATION
    NOINHERIT;

GRANT CONNECT ON DATABASE postgres TO nirf_readonly;
GRANT USAGE ON SCHEMA public TO nirf_readonly;
GRANT SELECT ON TABLE public."NIRF" TO nirf_readonly;

ALTER ROLE nirf_readonly
    SET default_transaction_read_only = on;
ALTER ROLE nirf_readonly
    SET statement_timeout = '10s';
