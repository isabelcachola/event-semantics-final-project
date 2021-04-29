import xml.etree.ElementTree as ET
import os

class RemanData:
    def __init__(self, datadir):
        self.datadir = datadir

    def read_data(self, split):
        tree = ET.parse(os.path.join(self.datadir, 'reman', 'reman-version1.0.xml'))
        root = tree.getroot()
        doc_ids = open(f'{split}_ids.txt').readlines()
        doc_ids = [d.strip() for d in doc_ids]
        data = []
        for doc in root:
            if doc.attrib['doc_id'] in doc_ids:
                text = doc[0].text

