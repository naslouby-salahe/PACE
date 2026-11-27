import ast
import re

from pace.types import ArtifactPayload, ClassIndex, IdentifierName, NameSet

REFERENCE_TEXT = re.compile(r"[\w.\[\], |]+")
IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


def _referenced_names(tree: ast.AST) -> NameSet:
    names: NameSet = set()
    for node in ast.walk(tree):
        match node:
            case ast.Name(id=name) | ast.Attribute(attr=name):
                names.add(IdentifierName(name))
            case ast.MatchClass(kwd_attrs=attributes):
                names.update(IdentifierName(attribute) for attribute in attributes)
            case ast.alias(name=dotted):
                names.update(IdentifierName(part) for part in dotted.split("."))
            case ast.Constant(value=str() as text) if REFERENCE_TEXT.fullmatch(text):
                names.update(IdentifierName(found) for found in IDENTIFIER.findall(text))
            case _:
                continue
    return names


def _is_plain_property(member: ast.stmt) -> bool:
    return (
        isinstance(member, ast.FunctionDef)
        and len(member.decorator_list) == 1
        and isinstance(member.decorator_list[0], ast.Name)
        and member.decorator_list[0].id == property.__name__
    )


def _pruned(node: ast.ClassDef, names: NameSet) -> ast.ClassDef:
    members = [
        member
        for member in node.body
        if not _is_plain_property(member)
        or (isinstance(member, ast.FunctionDef) and IdentifierName(member.name) in names)
    ]
    return ast.ClassDef(
        name=node.name,
        bases=node.bases,
        keywords=node.keywords,
        body=members or [ast.Pass()],
        decorator_list=node.decorator_list,
        type_params=node.type_params,
    )


def scientific_types_payload(
    types_source: ArtifactPayload, scope_sources: tuple[ArtifactPayload, ...]
) -> ArtifactPayload:
    module = ast.parse(types_source)
    classes: ClassIndex = {
        IdentifierName(node.name): node for node in module.body if isinstance(node, ast.ClassDef)
    }
    names: NameSet = set()
    for source in scope_sources:
        names |= _referenced_names(ast.parse(source))
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            names |= _referenced_names(node)
    while True:
        grown = names.union(
            *(
                _referenced_names(_pruned(node, names))
                for name, node in classes.items()
                if name in names
            )
        )
        if grown == names:
            break
        names = grown
    body = [
        _pruned(node, names) if isinstance(node, ast.ClassDef) else node
        for node in module.body
        if not isinstance(node, ast.ClassDef) or IdentifierName(node.name) in names
    ]
    return ArtifactPayload(
        ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=True).encode()
    )
