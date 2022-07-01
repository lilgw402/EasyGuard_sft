#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test visual tokenizer
'''

import os
import unittest
from unittest import TestCase
import pytest

import torch
from PIL import Image
from torchvision import transforms

from fex.nn.visual_tokenizer.base import create_visual_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../ci/test_data")


class TestResnet(TestCase):
    """ test create_visual_tokenizer """

    @unittest.skip(reason="skip")
    def setUp(self):
        input_image = Image.open(os.path.join(TEST_DIR, "dog.jpg"))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        # create a mini-batch as expected by the model
        self.input_batch = input_tensor.unsqueeze(0)

        self.model = resnet50()
        if int(os.environ.get('ISCI', 0)) != 1:
            print('loading pretrain model')
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=False))
        self.model.eval()
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

    @unittest.skip(reason="skip")
    def test_resnet_random(self):
        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = torch.randn([1, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = self.model(fake_input)['cls_score']
            logits = torch.nn.functional.softmax(output[0], dim=0)
            label = torch.argmax(logits)
            # self.assertEqual(label, 490)
            # self.assertEqual(label, 937)
            self.assertEqual(list(logits.shape), [1000])

    @unittest.skip(reason="skip")
    @pytest.mark.skipif(int(os.environ.get('ISCI', 0)) == 1, reason="CI environment not support")
    def test_resnet_dog(self):
        torch.manual_seed(42)
        with torch.no_grad():
            output = self.model(self.input_batch)['cls_score']
            logits = torch.nn.functional.softmax(output[0], dim=0)
            self.assertEqual(list(logits.shape), [1000])
            label = torch.argmax(logits).cpu().int()
            self.assertEqual(label, 258)  # Samoyed, Samoyede


if __name__ == '__main__':
    unittest.main()
