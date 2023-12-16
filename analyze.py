#!/usr/bin/env python3
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Optional
from tree_sitter_languages import get_parser
from openai import OpenAI


def change_prompt(diff: str, classes: List[str]) -> str:
    """ Prompt for asking for the API changes"""
    return f"""
Please list the abstract changes to the following classes,
List the abstract changes to the {', '.join(classes)} API's methods 
and properties that caused the following changes in the test code, 
in JSON of the form 
'{{"class": class name, "old": old method signature
or property, "new": new method signature or property, 
"decription": description of the change, 
"example": {{"old": old code, "new": new code}}}}': 


{diff}


in JSON, be as abstract and concise as possible, 
generalize wherever possible to keep the list small
"""


BASEDIR = Path(__file__).resolve().parent
JDK_DIR = BASEDIR / "jdk"


class WithRevision:
    """ bring the jdk folder to the revision """

    def __init__(self, revision: str):
        self.revision: str = revision
        self.old_revision: Optional[str] = None

    def __enter__(self):
        self.old_revision = subprocess.check_output(
            "git rev-parse HEAD",
            cwd=JDK_DIR, shell=True).decode("utf-8").strip()
        subprocess.check_call(f"git checkout {self.revision}",
                              cwd=JDK_DIR, shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        subprocess.check_call(f"git checkout {self.old_revision}",
                              cwd=JDK_DIR, shell=True,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)


tree_sitter_parser = get_parser("java")


def public_classes_in_file(file: Path) -> List[str]:
    """ returns a list of all public classes in a Java file """
    content = file.read_text()
    classes = []

    def modifiers(decl_node) -> Set[str]:
        return set(cc.text.decode() for child in decl_node.children if
                   child.type == "modifiers" for cc in child.children)

    def name(decl_node) -> str:
        return [child.text.decode() for child in decl_node.children
                if child.type == "identifier"][0]

    def body(decl_node):
        return [child for child in decl_node.children
                if child.type in ["class_body", "interface_body"]][0]

    def recursive_walk(node, parent: str = "",
                       is_parent_interface: bool = False):
        is_interface = node.type == "interface_declaration"
        is_class = node.type == "class_declaration"
        if not is_interface and not is_class:
            return
        mods = modifiers(node)
        if not is_parent_interface and not is_interface:
            if "public" not in mods:
                return
        combined_name = f"{parent}.{name(node)}" \
            if parent else name(node)
        classes.append(combined_name)
        for child in body(node).children:
            recursive_walk(child, combined_name, is_interface)

    tree = tree_sitter_parser.parse(content.encode("utf-8"))
    for n in tree.root_node.children:
        recursive_walk(n)
    return classes


def public_classes(revision: str, file_pattern: str) -> List[str]:
    """
    Returns a list of all public classes and interfaces in the JDK
    :param revision:
    :param file_pattern:
    :return:
    """
    with WithRevision(revision):
        classes = []
        for file in JDK_DIR.glob(file_pattern):
            if file.is_dir():
                classes.extend(
                    public_classes(revision, f"{file}/*"))
                continue
            if file.suffix != ".java":
                continue
            classes.extend(public_classes_in_file(file))
        return classes


@dataclass
class Example:
    old: str
    new: str

    def to_json(self):
        return {
            "old": self.old,
            "new": self.new
        }


@dataclass
class EntityDiff:
    old: str
    new: str
    description: str
    example: Example

    def to_json(self, class_name: str) -> dict:
        return {
            "class": class_name,
            "old": self.old,
            "new": self.new,
            "description": self.description,
            "example": self.example.to_json()
        }

    @staticmethod
    def from_json(json: dict) -> 'EntityDiff':
        return EntityDiff(json["old"], json["new"],
                          json["description"],
                          Example(json["example"]["old"],
                                  json["example"]["new"]))


@dataclass
class ClassDiff:
    name: str
    entity_diffs: List[EntityDiff]

    def to_json(self) -> list:
        return [e.to_json(self.name) for e in self.entity_diffs]

    @staticmethod
    def from_json(json: list) -> 'ClassDiff':
        assert (len(json) > 0 and
                all(json[0]["class"] == e["class"] for e in json))
        return ClassDiff(json[0]["class"],
                         [EntityDiff.from_json(e) for e in json])


@dataclass
class AnalyzedDiff:
    class_diffs: List[ClassDiff]
    summary: Optional[str] = None

    def to_json(self):
        return [c.to_json() for c in self.class_diffs]

    @staticmethod
    def from_json(json: list,
                  summary: Optional[str]) -> 'AnalyzedDiff':
        # collect JSON for each class
        class_json = {}
        for class_json in json:
            class_name = class_json[0]["class"]
            class_json[class_name] = class_json
        # create ClassDiff objects
        class_diffs = []
        for class_name, class_json in class_json.items():
            class_diffs.append(ClassDiff.from_json(class_json))
        return AnalyzedDiff(class_diffs, summary)


def openai_key() -> str:
    key_file = BASEDIR / ".openai.key"
    if not key_file.exists():
        print("Please store your OpenAI API key in .openai.key",
              file=sys.stderr)
        sys.exit(1)
    return key_file.read_text().strip()


def analyze_diff(diff: str, classes: List[str]) -> AnalyzedDiff:
    prompt = change_prompt(diff, classes)
    print(prompt)
    client = OpenAI(api_key=openai_key())
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    print(repr(chat_completion))


def get_diff(ref1: str, ref2: str, file: Path,
             remove_license_diff: bool = True) -> str:
    diff = subprocess.check_output(
        f"git diff {ref1}..{ref2} '{file}'",
        cwd=JDK_DIR, shell=True).decode("utf-8").strip()
    if remove_license_diff:
        lines = diff.split("\n")
        new_lines = []
        found_copyright = False
        found_end_of_license = False
        for line in lines:
            if found_end_of_license:
                new_lines.append(line)
                continue
            if "Copyright (c)" in line:
                found_copyright = True
            if not found_copyright:
                new_lines.append(line)
                continue
            if not line[2] == "*":
                found_end_of_license = True
                new_lines.append(line)
        diff = "\n".join(new_lines)
    return diff


def analyze_ref_diffs(ref1: str, ref2: str,
                      test_file_pattern: str,
                      api_file_pattern: str) -> List[AnalyzedDiff]:
    classes = list(set(public_classes(ref1, api_file_pattern) +
                       public_classes(ref2, api_file_pattern)))
    res: List[AnalyzedDiff] = []
    # analyze all test files
    for file in JDK_DIR.glob(test_file_pattern):
        if file.is_dir():
            continue
        if file.suffix != ".java":
            continue
        res.append(analyze_diff(get_diff(ref1, ref2, file), classes))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_classes_parser = subparsers.add_parser("list_classes")
    list_classes_parser.add_argument("ref")
    list_classes_parser.add_argument("file_pattern")
    list_classes_parser.add_argument("--sep", default=", ")
    analyze_diff_parser = subparsers.add_parser("analyze_diff")
    analyze_diff_parser.add_argument("diff_file", type=Path)
    analyze_diff_parser.add_argument("classes", nargs="*")
    analyze_ref_parser = subparsers.add_parser("analyze_ref")
    analyze_ref_parser.add_argument("ref1")
    analyze_ref_parser.add_argument("ref2")
    analyze_ref_parser.add_argument("test_file_pattern")
    analyze_ref_parser.add_argument("api_file_pattern")
    analyze_panama_parser = subparsers.add_parser("analyze_panama")
    analyze_panama_parser.add_argument("ref1")
    analyze_panama_parser.add_argument("ref2")
    args = parser.parse_args()
    # execute command
    if args.command == "list_classes":
        print(args.sep.join(
            public_classes(args.ref, args.file_pattern)))
    elif args.command == "analyze_diff":
        analyze_diff(args.diff_file, args.classes)
    elif args.command == "analyze_ref":
        analyze_ref_diffs(args.ref1, args.ref2,
                          args.test_file_pattern,
                          args.api_file_pattern)
    elif args.command == "analyze_panama":
        analyze_ref_diffs(args.ref1, args.ref2,
                          "test/jdk/java/foreign/**/*.java",
                          "src/java.base/share/classes/java/lang/foreign/"
                          "**/*.java")
