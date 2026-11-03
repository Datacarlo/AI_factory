import tempfile
import unittest
from pathlib import Path
from tools.datasets.catalog_layout import physical_path as logical_path
from tools.datasets.artifact_publication import contained_path as physical_path


class PublicationLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make(self, relative):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{}')
        return p

    def test_plain_public_layout_unchanged(self):
        p = self.make('datasets/model/heston.json')
        self.assertEqual(physical_path(self.root, 'datasets/model/heston.json'), p)

    def test_published_reads_and_pending_replacements(self):
        p = self.make('datasets/published/model/heston.json')
        self.assertEqual(physical_path(self.root, 'datasets/model/heston.json'), p)
        u = self.make('datasets/unpublished/model/heston.json')
        self.assertEqual(physical_path(self.root, 'datasets/model/heston.json'), u)
        self.assertEqual(physical_path(self.root, 'datasets/published/model/heston.json'), p)

    def test_new_dataset_goes_to_pending(self):
        (self.root / 'datasets/published').mkdir(parents=True)
        self.assertEqual(physical_path(self.root, 'datasets/model/new.json'),
                         self.root / 'datasets/unpublished/model/new.json')

    def test_codegen_keeps_canonical_paths(self):
        self.make('datasets/published/model/heston.json')
        self.assertEqual(logical_path(self.root, 'datasets/model/heston.json'),
                         self.root / 'datasets/model/heston.json')

    def test_campaign_work_layout_unchanged(self):
        (self.root / 'datasets/published').mkdir(parents=True)
        p = self.make('work/datasets/model/heston.json')
        self.assertEqual(physical_path(self.root, 'work/datasets/model/heston.json'), p)

    def test_traversal_still_rejected(self):
        for path in ('../dataset.json', '/tmp/dataset.json'):
            with self.assertRaises(ValueError):
                physical_path(self.root, path)

    def test_publication_symlink_still_rejected(self):
        p = self.make('datasets/published/model/heston.json')
        q = self.root / 'datasets/unpublished/model/heston.json'
        q.parent.mkdir(parents=True)
        q.symlink_to(p)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            physical_path(self.root, 'datasets/model/heston.json')

    def test_canonical_categories_resolve_without_changing_generator_paths(self):
        (self.root / 'datasets/published').mkdir(parents=True)
        for name in ('parameters/model/heston/parameters/base.json',
                     'parameters/product/european_option/calls.json',
                     'parameters/curve/flat/base.json',
                     'parameters/correlation/multi/base.json',
                     'samples/equity/rough/rough_bergomi/samples_01.json',
                     'single_asset/fixed_income/cir/prices/call.json',
                     'multi_asset/basket_option/basket.json'):
            with self.subTest(name=name):
                p = self.make('datasets/unpublished/' + name)
                self.assertEqual(physical_path(self.root, 'datasets/' + name), p)
                self.assertEqual(logical_path(self.root, 'datasets/' + name),
                                 self.root / 'datasets' / name)
