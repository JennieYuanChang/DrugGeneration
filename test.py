from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile
import json
import numpy as np
from six.moves import urllib
import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import Response
from flask import request
import urllib.request

app = Flask(__name__)

@app.route("/predict")
def predict():
#example data:
    url = request.args.get('url', default = "", type = str)
    urllib.request.urlretrieve(url, "image.jpg")
    js = cls.main("image.jpg")
#then do this
    return Response(json.dumps(js),  mimetype='application/json')
   

