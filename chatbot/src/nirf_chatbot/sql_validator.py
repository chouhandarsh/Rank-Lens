import re

import sqlglot
from sqlglot import exp


_DOLLAR_QUOTE = re.compile(r"\$[A-Za-z_][A-Za-z_0-9]*\$|\$\$")


def strip_sql_comments(query: str) -> str:
    output = []
    index = 0
    block_depth = 0
    quote = None
    dollar_quote = None

    while index < len(query):
        if block_depth:
            if query.startswith("/*", index):
                block_depth += 1
                index += 2
            elif query.startswith("*/", index):
                block_depth -= 1
                index += 2
            else:
                index += 1
            continue

        if dollar_quote:
            if query.startswith(dollar_quote, index):
                output.append(dollar_quote)
                index += len(dollar_quote)
                dollar_quote = None
            else:
                output.append(query[index])
                index += 1
            continue

        if quote:
            char = query[index]
            output.append(char)
            index += 1
            if char == quote:
                if index < len(query) and query[index] == quote:
                    output.append(query[index])
                    index += 1
                else:
                    quote = None
            elif char == "\\" and quote == "'" and index < len(query):
                output.append(query[index])
                index += 1
            continue

        if query.startswith("--", index):
            newline = query.find("\n", index + 2)
            if newline == -1:
                output.append(" ")
                break
            output.extend((" ", "\n"))
            index = newline + 1
        elif query.startswith("/*", index):
            output.append(" ")
            block_depth = 1
            index += 2
        elif query[index] in {"'", '"'}:
            quote = query[index]
            output.append(query[index])
            index += 1
        elif query[index] == "$":
            match = _DOLLAR_QUOTE.match(query, index)
            if match:
                dollar_quote = match.group()
                output.append(dollar_quote)
                index = match.end()
            else:
                output.append(query[index])
                index += 1
        else:
            output.append(query[index])
            index += 1

    return "".join(output)


def _statement_count(query: str) -> int:
    count = 0
    has_content = False
    index = 0
    quote = None
    dollar_quote = None

    while index < len(query):
        if dollar_quote:
            if query.startswith(dollar_quote, index):
                index += len(dollar_quote)
                dollar_quote = None
            else:
                index += 1
            has_content = True
            continue

        if quote:
            char = query[index]
            index += 1
            if char == quote:
                if index < len(query) and query[index] == quote:
                    index += 1
                else:
                    quote = None
            elif char == "\\" and quote == "'" and index < len(query):
                index += 1
            has_content = True
            continue

        char = query[index]
        if char in {"'", '"'}:
            quote = char
            has_content = True
            index += 1
        elif char == "$":
            match = _DOLLAR_QUOTE.match(query, index)
            if match:
                dollar_quote = match.group()
                has_content = True
                index = match.end()
            else:
                has_content = True
                index += 1
        elif char == ";":
            if has_content:
                count += 1
                has_content = False
            index += 1
        else:
            if not char.isspace():
                has_content = True
            index += 1

    return count + int(has_content)


def prepare_sql(query: str) -> str:
    prepared = strip_sql_comments(query).strip()
    if prepared.endswith(";"):
        prepared = prepared[:-1].rstrip()
    return prepared


def normalize_sql(query: str) -> str:
    query = prepare_sql(query)
    if not query:
        return query

    try:
        parsed = sqlglot.parse(query, read="postgres")
    except sqlglot.errors.SqlglotError:
        return query

    if len(parsed) != 1 or not isinstance(parsed[0], exp.Query):
        return query

    statement = parsed[0]
    if statement.args.get("limit") is None:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(1000)))
    return statement.sql(dialect="postgres")


def validate_sql(query: str) -> bool:
    prepared = prepare_sql(query)
    if not prepared:
        raise ValueError("Invalid SQL: output is empty")
    if _statement_count(prepared) > 1:
        raise ValueError("Invalid SQL: multiple SQL statements are not allowed")
    return True
