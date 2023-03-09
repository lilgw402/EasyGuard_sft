import json

di = """
    {
    "page_info": {
        "tag_title": "test",
        "core_content": "test"
    }
}
"""

aa = json.loads(di)

print(aa['page_info'])