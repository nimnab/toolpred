import json
import os

import dropbox
import inflect
import requests

base = "https://mapi.yummly.com/mapi/v17/content/feed?&start={}&limit=20"

os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = "172.26.0.47:8118"
os.environ['https_proxy'] = os.environ['HTTPS_PROXY'] = "172.26.0.44:8118"

inflect = inflect.engine()
def singular(word):
    """
    Returns a singularized word
    :rtype: str
    """
    head = inflect.singular_noun(word)
    if not head:
        head = word
    return head.lower().strip()

def upload(file):
    # Create a dropbox object using an API v2 key
    d = dropbox.Dropbox('YKzgZf7GQKUAAAAAAAAMxT74UQMneilzVkUqpb07fdthP1Vu_rdPgzwZZ_XO_ayK')
    targetfile = '/Results/' + file.split('/')[-1]
    # open the file and upload it
    with open(file, "rb") as f:
        # upload gives you metadata about the file
        # we want to overwite any previous version of the file
        meta = d.files_upload(f.read(), targetfile, mode=dropbox.files.WriteMode("overwrite"))

# @atexit.register
# def exithandler():
#     with open('/hri/localdisk/nnabizad/yummlyfull.json', 'w+') as f:
#         json.dump(jsons, f)

def scrap_data():
    jsons= []
    names = set()
    for i in range(0,50000,19):
        try:
            r = requests.get(base.format(i))
            jslist = r.json()['feed']
            # print(len(jslist))
            if i % 10 == 0:
                print('{}, {}, {}'.format(str(i),len(jsons), len(names)))

            for js in jslist:
                # name = js['display']['displayName'].lower().strip()
                if js['display']['source']:
                    url = js['display']['source']['sourceRecipeUrl']
                    if url not in names:
                        names.add(url)
                        jsons.append(js)
                # else:
            #     print(i, name)
        except Exception as e:
            print('error',e)
            continue

    with open('/hri/localdisk/nnabizad/yummlyfull.json', 'w+') as f:
        json.dump(jsons, f)

    upload('/hri/localdisk/nnabizad/yummlyfull.json')

def clean(txt):
    if 'x13' in txt:txt = '9x13-inch baking dish'
    if 'tooth' in txt:txt = 'toothpick'
    if txt == 'sauce pan': txt = 'saucepan'
    if txt == 'sauté pan': txt = 'saute pan'
    if txt == 'springform pan': txt = 'spring form pan'
    return(singular(txt.lower().strip()))


def extract(file):
    recsings = []
    rectools = []
    titles = []
    allings = []
    alltools = []
    for rec in file:
        _tmpeq = []
        _tmping = []
        if 'guidedVariations' in rec['content'].keys():
            for step in rec['content']['guidedVariations'][0]['actions']:
                if 'equipment' in step['stepGroups'][0]['steps'][0]:
                    eqs = tuple([clean(i['name']) for i in step['stepGroups'][0]['steps'][0]['equipment']])
                    ings = tuple(
                        [singular(i['ingredient']) for i in step['stepGroups'][0]['steps'][0]['ingredientLines']])
                    [allings.append(i) for i in ings]
                    [alltools.append(i) for i in eqs]
                    _tmpeq.append(eqs)
                    _tmping.append(ings)
            if len(_tmpeq) > 0 or len(_tmping) >0:
                recsings.append(_tmping)
                rectools.append(_tmpeq)
                titles.append(rec['display']['displayName'])
    return recsings,  rectools, titles


if __name__ == '__main__':
    with open('/hri/localdisk/nnabizad/yummlyfull.json', 'r+') as f:
        file = json.load(f)
    recsings, rectools, titles = extract(file)
    print()

