import requests

url = "https://api.github.com/graphql"
headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}
query = """
{
  repository(owner: "loyaloy", name: "vuedemo1") {
    object(expression: "master:src/router.js") {
      ... on Blob {
        text
      }
    }
  }
}
"""
response = requests.post(url, json={"query": query}, headers=headers)
if response.status_code == 200:
    data = response.json()
    file_content = data["data"]["repository"]["object"]["text"]
    with open("router.js", "w") as file:
        file.write(file_content)
    print("Dosya indirildi!")
else:
    print("Hata:", response.text)