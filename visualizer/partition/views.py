from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from .apps import PartitionConfig

import sys
sys.path.append("../..")
import graph
from optimizer.dlrm_optimizer import get_stage_statistics


# Create your views here.

def index(request):
    dotString = open(PartitionConfig.defaultDotFile, 'r').read()
    txtString = open(PartitionConfig.defaultTxtFile, 'r').read()
    contents = {
        'defaultDotFile': dotString,
        'defaultTxtFile': txtString
    }
    return render(request, "index.html", contents)

def submitTxt(request):
    txtString = request.GET.get("txt", "")
    txtFile = open(PartitionConfig.defaultTxtFile, 'w')
    txtFile.write(txtString)
    txtFile.close()

    gr = graph.Graph.from_str(txtString)
    gr.to_dot(PartitionConfig.defaultDotFile)
    dotString = open(PartitionConfig.defaultDotFile).read()

    #print(txtString)
    results = {
        "dot": dotString,
        "statistics": get_stage_statistics(gr)
    }

    return JsonResponse(results)
