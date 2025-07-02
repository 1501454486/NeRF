import yaml
import os
import sys
from pprint import pprint

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import cfg, args
from src.models.kilonerf.renderer.sampler import Sampler


def main():
    sampler = Sampler()


if __name__ == "__main__":
    main()