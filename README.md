# API change discovery

The aim is to find changes in the [Panama API](https://openjdk.org/projects/panama/) of the OpenJDK between two versions,
by examining the test diffs using GPT 3.5.

The idea is that a change like this:

```diff
         int strlen(String msg) throws Throwable {
             try (var arena = Arena.ofConfined()) {
-                MemorySegment s = arena.allocateUtf8String(msg);
+                MemorySegment s = arena.allocateString(msg);
                 return (int)strlen.invokeExact(s);
             }
         }
```

Is caused by a change in the API, and that the diff can be used to find the change.
In this case, the change is that method `Arena.allocateUtf8String` was renamed to `Arena.allocateString`
(with an optional charset parameter that seems to be `StandardCharsets.UTF_8` by default).

This is of course only an approximation, but it should be good enough to find most changes
and be a good starting point for further investigation.

AI comes into play because we're doing the analysis on the test diffs automically.
We use the following query:

```
List the abstract changes to the <possibly used classes> API's methods 
and properties that caused the following changes in the test code, 
in JSON of the form '{"class": class name, "old": old method signature 
or property, "new": new method signature or property, 
"decription": description of the change, "example": {"old": old code, "new": new code}}': 


<diff>

in JSON, be as abstract and concise as possible, generalize whereever possible to keep the list small
```

Running this query on the diff above, we get a result like the following 
(actually computed by GitHub Copilot while writing this README):

```json
[
    {
        "class": "Arena",
        "old": "allocateUtf8String(String)",
        "new": "allocateString(String)",
        "description": "method allocateUtf8String was renamed to allocateString",
        "example": {
            "old": "MemorySegment s = arena.allocateUtf8String(msg);",
            "new": "MemorySegment s = arena.allocateString(msg);"
        }
    }
]
```

I tried a few local large language models, like Llama2, but the results were not very good.

## Setup

- make sure to have all submodules ready (and update them if necessary)
  - use `--recursive` when cloning
  - or `git submodule update --init --recursive` after cloning
- install [openai python package](https://pypi.org/project/openai/) and optionally [tree-sitter-languages](https://pypi.org/project/tree-sitter-languages/)
  - `pip3 install openai tree-sitter-languages`
- store your API key in `.openai.key` in the root directory

## Usage

```shell
# list possible API class names
python3 analyze.py list_classes <ref> <file pattern>

# analyze a single diff file for changes in the passed classes
python3 analyze.py analyze_diff <diff file> <class names>

# analyze the diff between the test files of two refs or tags
python3 analyze.py analyze_ref <ref 1> <ref 2> <test file pattern> <API file pattern>

# analyze the panama diff
python3 analyze.py analyze <ref> <commit or tag 2>
# equivalent to
python3 analyze.py analyze_ref <ref 1> <ref 2> "test/jdk/java/foreign/**/*.java" "src/java.base/share/classes/java/lang/foreign/**/*.java"
```

Example usage:

```shell
# list panama API classes
./analyze.py list_classes HEAD "src/java.base/share/classes/java/lang/foreign/**/*.java"

# analyze the diff of a single test file
./analyze.py analyze_ref jdk-21-ga HEAD "test/micro/org/openjdk/bench/java/lang/foreign/StrLenTest.java" "src/java.base/share/classes/java/lang/foreign/**/*.java"
```

TODO:

- [ ] test with OpenAI key
- [ ] summarize results
- [ ] generate markdown report (and transform to html)
- [ ] improve README (fix LICENSE statement, ...)
- [ ] start blog post

## License
MIT