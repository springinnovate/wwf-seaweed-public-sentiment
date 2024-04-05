import glob

AQUACULTURE_KEYWORDS = {
    "aquaculture",
    "marine aquaculture",
    "offshore aquaculture"
    }

SEAWEED_AQUACULTURE = {
    'seaweed',
    'kelp',
    'sea moss',
    'aquaculture',
    'farm',
    'cultivat',
}

def main():
    for json_file in glob.glob(
            'data/successfulresults_seaweed_regional_search/*.json'):
        with open(json_file, encoding='utf-8') as file:
            in_article = False
            for line in file:
                line = line.strip()
                if not in_article:
                    if line.startswith('"title"'):
                        in_article = True
                else:
                    if line.startswith('"paragraph"'):
                        if any(
                                keyword in line
                                for keyword in AQUACULTURE_KEYWORDS):
                            print(f'AQUACUTLURE: {line}')
                        # if any(
                        #         keyword in line
                        #         for keyword in SEAWEED_AQUACULTURE):
                        #     print(f'SEAWEED AQUACUTLURE: {line}')


if __name__ == '__main__':
    main()
