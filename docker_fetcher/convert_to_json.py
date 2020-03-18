import bs4
import lxml
import re
import glob
import json
from multiprocessing import Pool
import os
import sys
import jsonlines

dateReg = re.compile('(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})')


# load the xml file in beutiful soup. make sure you use the lxml-xml parser
reg = re.compile('([\dE-]+)')

def rule_to_dict(rule,_type,year,month,day):
    '''
    Gets a rule, and its type (rule/prorule), return a dictionary with the metadata we need and list of
    paragraphras
    '''
    preamble = rule.PREAMB
    try:
        doc_id = reg.search(rule.FRDOC.text).group(0)
    except:
        print(rule.FRDOC.text)
    url = 'https://www.federalregister.gov/documents/full_text/html/{year}/{day}/{month}/{docId}.html'.format(
    year=year,month=month,day=day,docId=doc_id)
    return {
        'type':_type,
        'agency':preamble.AGENCY.text,
        'title':preamble.SUBJECT.text,
        'doc_id':doc_id , #Use this as an aggregation code
        'text':rule.text,
        'url':url
    }
def doFile(fname):
    results = []
    fname
    soup = bs4.BeautifulSoup(open(fname), "lxml-xml")
    # This is how we extract rules and proposed rules
    date = dateReg.search(fname).groupdict()
    if soup.FEDREG.RULES:
        rules = soup.FEDREG.RULES.findAll('RULE')

        ruleResults = map(lambda x: rule_to_dict(x, 'rule', date['year'], date['month'], date['day']), rules)
        results = list(ruleResults)
    else:
        print(fname)
    if soup.FEDREG.PRORULES:
        proposedRules = soup.FEDREG.PRORULES.findAll("PRORULE")
        proRuleResults = map(lambda x: rule_to_dict(x, 'prorule', date['year'], date['month'], date['day']),
                             proposedRules)

        # Combine them into one list. map is lazy, so we need to call list on each
        results += list(proRuleResults)
    return results
if __name__ =='__main__':
    fname = sys.argv[1]
    results = doFile(fname)
    outname = os.path.join('/results/',os.path.basename(fname)+'.jsonl')
    with jsonlines.open(outname, mode='w') as writer:
        for row in results:
            writer.write(row)
