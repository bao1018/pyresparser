import os
import argparse
from pprint import pprint
import io
import multiprocessing as mp
import urllib
from urllib.request import Request, urlopen
from pyresparser import ResumeParser

def get_remote_data():
    try:
        remote_file = 'https://www.omkarpathak.in/downloads/OmkarResume.pdf'
        print('Extracting data from: {}'.format(remote_file))
        req = Request(remote_file, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        _file = io.BytesIO(webpage)
        _file.name = remote_file.split('/')[-1]
        resume_parser = ResumeParser(_file)
        return [resume_parser.get_extracted_data()]
    except urllib.error.HTTPError:
        return 'File not found. Please provide correct URL for resume file.'

def get_local_data():
    data = ResumeParser('./resumes/Aaron-Gocer.pdf').get_extracted_data()
    # data = ResumeParser('./resumes/john_doe.pdf').get_extracted_data()
    # data = ResumeParser('./resumes/wayne_li.pdf').get_extracted_data()
    # data = ResumeParser('OmkarResume.pdf').get_extracted_data()
    # data = ResumeParser('./resumes/bruce_wayne_fullstack.pdf').get_extracted_data()
    # data = ResumeParser('./resumes/harvey_dent_mle.pdf').get_extracted_data()
    return data
 
result = get_local_data()
print(result)
