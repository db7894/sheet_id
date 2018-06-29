# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

__sets = {}
sys.path.insert(0,'/data1/dbashir/Project/Summer2018/DeepWatershedDetection/lib/datasets/')
from pascal_voc import pascal_voc
from coco import coco
from deep_scores import deep_scores
from musicma import musicma

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up musical dataset
for year in ['2017']:
  for split in ['train', 'val', 'test', 'debug','train100','train10000', 'test100',]:
    name = 'DeepScores_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: deep_scores(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

  # Set up for Musicma++
  for year in ['2017']:
    for split in ['train', 'test', 'val']:
      name = 'MUSICMA++_{}_{}'.format(year, split)
      __sets[name] = (lambda split=split, year=year: musicma(split, year))

# Set up coco_2014_<split>
for year in ['2017']:
  for split in ['train', 'val']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))



def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
