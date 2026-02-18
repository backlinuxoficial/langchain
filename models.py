from google import genai
client = genai.Client(api_key="AIzaSyCyinO-I73mjv0EJEdf-VPF_G4tM17_R84")
for m in client.models.list():
    print(m.name, m.supported_actions)