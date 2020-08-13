from django.apps import AppConfig
from django.conf import settings
import os

class PartitionConfig(AppConfig):
    name = 'partition'
    APP_DIR = os.path.join(settings.BASE_DIR, name)
    defaultDotFile = os.path.join(APP_DIR, 'static/tmp.dot')
    defaultTxtFile = os.path.join(APP_DIR, 'static/tmp.txt')
