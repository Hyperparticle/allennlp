from allennlp.common.testing.model_test_case import ModelTestCase
import json


class UDifyMultilangUDParserTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            self.PROJECT_ROOT / "training_config" / "udify_multilang_ud_parser.json",
            self.FIXTURES_ROOT / "data" / "dependencies_multilang" / "*",
        )

    def test_dependency_parser_can_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.param_file,
            overrides=json.dumps({
                "train_data_path": "/home/hyper/Documents/repos/allennlp/allennlp/tests/fixtures/data/dependencies_multilang/*",
                "validation_data_path": "/home/hyper/Documents/repos/allennlp/allennlp/tests/fixtures/data/dependencies_multilang/*",
                "test_data_path": "/home/hyper/Documents/repos/allennlp/allennlp/tests/fixtures/data/dependencies_multilang/*",
            })
        )
