import requests
import json
import atexit
base = "https://mapi.yummly.com/mapi/v17/content/feed?tag=list.recipe.other.guided&start={}&limit=10"



@atexit.register
def exithandler():
    with open('/hri/localdisk/nnabizad/yummlyf.json', 'w+') as f:
        json.dump(jsons, f)


jsons= []
names = set()
for i in range(0,50000,9):
    try:
        r = requests.get(base.format(i))
        jslist = r.json()['feed']
        for js in jslist:
            name = js['display']['displayName'].strip().lower()
            url = js['display']['source']['sourceRecipeUrl']
            if i %10 == 0: print(str(i))
            if url not in names:
                names.add(url)
                jsons.append(js)
            # else:
            #     print(i, name)
    except Exception as e:
        print(e)
        continue

with open('/hri/localdisk/nnabizad/yummlyf.json', 'w+') as f:
    json.dump(jsons, f)

