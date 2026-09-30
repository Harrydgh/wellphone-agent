from __future__ import annotations

import unittest

from wellphone_agent.agent.core import AgentError
from wellphone_agent.agent.shopping_intent import DeepSeekShoppingIntentParser


class FakeModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[object] = []

    def invoke(self, messages: object) -> object:
        self.calls.append(messages)
        return self.response


class ShoppingIntentTests(unittest.TestCase):
    def test_parses_first_product_default_specs_and_forces_cart_boundary(self) -> None:
        model = FakeModel(
            {
                "query": "鲜肉包子",
                "selection_strategy": "first_product",
                "quantity": 1,
                "specification_policy": "default_specs",
                "stop_at": "payment",
            }
        )
        intent = DeepSeekShoppingIntentParser(
            model="test", structured_model=model
        ).parse("搜索鲜肉包子，选第一个真实商品，默认规格，一份，加入购物车")
        self.assertEqual(intent.query, "鲜肉包子")
        self.assertEqual(intent.product_query, "鲜肉包子")
        self.assertIsNone(intent.merchant_query)
        self.assertEqual(intent.selection_strategy, "first_match")
        self.assertEqual(intent.specification_policy, "default")
        self.assertEqual(intent.quantity, 1)
        self.assertEqual(intent.stop_at, "cart")
        self.assertIn("不得生成结算", str(model.calls[0]))

    def test_accepts_common_store_and_product_aliases(self) -> None:
        model = FakeModel(
            {
                "intent": "add_to_cart",
                "search_keyword": "瑞幸咖啡",
                "product_name": "生椰拿铁",
                "selection_strategy": "exact_match",
                "quantity": 1,
                "specification_policy": "default",
                "stop_at": "cart",
            }
        )
        intent = DeepSeekShoppingIntentParser(
            model="test", structured_model=model
        ).parse("搜索瑞幸咖啡，选择生椰拿铁，默认规格，一份，加入购物车")
        self.assertEqual(intent.app, "meituan")
        self.assertEqual(intent.merchant_query, "瑞幸咖啡")
        self.assertEqual(intent.product_query, "生椰拿铁")

    def test_retries_invalid_structure_with_explicit_correction(self) -> None:
        class SequenceModel:
            def __init__(self) -> None:
                self.calls: list[object] = []

            def invoke(self, messages: object) -> object:
                self.calls.append(messages)
                if len(self.calls) == 1:
                    return {"merchant_name": "瑞幸咖啡"}
                return {
                    "merchant_query": "瑞幸咖啡",
                    "product_query": "生椰拿铁",
                }

        model = SequenceModel()
        intent = DeepSeekShoppingIntentParser(
            model="test", structured_model=model
        ).parse("帮我去瑞幸找一杯生椰拿铁，默认就行，放购物车")
        self.assertEqual(intent.product_query, "生椰拿铁")
        self.assertEqual(len(model.calls), 2)
        self.assertIn("上一次返回没有通过", str(model.calls[1]))

    def test_rejects_payment_request_before_model_call(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        parser = DeepSeekShoppingIntentParser(model="test", structured_model=model)
        with self.assertRaisesRegex(AgentError, "超出当前安全边界"):
            parser.parse("买一份鲜肉包子并立即支付")
        self.assertEqual(model.calls, [])

    def test_rejects_unsupported_quantity_from_model(self) -> None:
        model = FakeModel(
            {
                "query": "鲜肉包子",
                "selection_strategy": "first_match",
                "quantity": 2,
                "specification_policy": "default",
            }
        )
        parser = DeepSeekShoppingIntentParser(model="test", structured_model=model)
        with self.assertRaisesRegex(AgentError, "只支持加购 1 份"):
            parser.parse("两份鲜肉包子")
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
