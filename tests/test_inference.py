#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块单元测试

覆盖:
  - API 服务路由和请求格式
  - 批量推理输入输出格式
  - 问题加载（JSON/CSV）
  - 结果保存
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 测试: 批量推理 - 数据加载
# ============================================================
class TestBatchInferenceDataLoading:
    """测试批量推理的输入数据加载。"""

    def test_load_json_list_of_strings(self, tmp_path):
        """加载 JSON 字符串列表。"""
        from scripts.inference.batch_inference import load_questions

        data = ["问题1", "问题2", "问题3"]
        f = tmp_path / "questions.json"
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        result = load_questions(str(f))
        assert len(result) == 3
        assert result[0]["question"] == "问题1"

    def test_load_json_list_of_dicts(self, tmp_path):
        """加载 JSON 字典列表。"""
        from scripts.inference.batch_inference import load_questions

        data = [
            {"question": "什么是PE？", "task_type": "financial_qa"},
            {"question": "分析茅台", "task_type": "stock_analysis"},
        ]
        f = tmp_path / "questions.json"
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        result = load_questions(str(f))
        assert len(result) == 2
        assert result[1]["task_type"] == "stock_analysis"

    def test_load_csv(self, tmp_path):
        """加载 CSV 文件。"""
        from scripts.inference.batch_inference import load_questions

        f = tmp_path / "questions.csv"
        f.write_text("question,category\n什么是市盈率,qa\n分析贵州茅台,stock\n", encoding="utf-8")

        result = load_questions(str(f))
        assert len(result) == 2
        assert result[0]["question"] == "什么是市盈率"

    def test_unsupported_format(self, tmp_path):
        """不支持的文件格式应抛异常。"""
        from scripts.inference.batch_inference import load_questions

        f = tmp_path / "questions.txt"
        f.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="不支持的文件格式"):
            load_questions(str(f))

    def test_csv_missing_question_column(self, tmp_path):
        """CSV 缺少 question 列应报错。"""
        from scripts.inference.batch_inference import load_questions

        f = tmp_path / "bad.csv"
        f.write_text("text,label\ntest,1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="question"):
            load_questions(str(f))


# ============================================================
# 测试: 批量推理 - 结果保存
# ============================================================
class TestBatchInferenceResultSaving:
    """测试结果保存功能。"""

    def test_save_json(self, tmp_path):
        """保存 JSON 结果。"""
        from scripts.inference.batch_inference import save_results

        results = [
            {"question": "Q1", "answer": "A1", "input_tokens": 10, "output_tokens": 20},
            {"question": "Q2", "answer": "A2", "input_tokens": 15, "output_tokens": 30},
        ]
        out_file = str(tmp_path / "results.json")
        save_results(results, out_file)

        with open(out_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert len(saved) == 2
        assert saved[0]["answer"] == "A1"

    def test_save_csv(self, tmp_path):
        """保存 CSV 结果。"""
        import pandas as pd
        from scripts.inference.batch_inference import save_results

        results = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ]
        out_file = str(tmp_path / "results.csv")
        save_results(results, out_file)

        df = pd.read_csv(out_file, encoding="utf-8-sig")
        assert len(df) == 2
        assert "answer" in df.columns

    def test_save_creates_parent_dirs(self, tmp_path):
        """保存时自动创建父目录。"""
        from scripts.inference.batch_inference import save_results

        results = [{"question": "Q", "answer": "A"}]
        out_file = str(tmp_path / "nested" / "dir" / "results.json")
        save_results(results, out_file)
        assert Path(out_file).exists()


# ============================================================
# 测试: API Server 请求模型
# ============================================================
class TestAPIRequestModels:
    """测试 API Server 的 Pydantic 模型。"""

    def test_chat_completion_request_defaults(self):
        """ChatCompletionRequest 默认值。"""
        from scripts.inference.api_server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="你好")]
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.stream is False
        assert req.model == "fin-instruct"

    def test_chat_completion_request_custom(self):
        """ChatCompletionRequest 自定义参数。"""
        from scripts.inference.api_server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            model="custom",
            messages=[
                ChatMessage(role="system", content="系统提示"),
                ChatMessage(role="user", content="问题"),
            ],
            temperature=0.3,
            max_tokens=512,
            stream=True,
        )
        assert req.temperature == 0.3
        assert req.max_tokens == 512
        assert req.stream is True
        assert len(req.messages) == 2


# ============================================================
# 测试: API Server 路由
# ============================================================
class TestAPIRoutes:
    """测试 API Server 路由。"""

    @pytest.fixture
    def client(self):
        """创建测试客户端。"""
        from fastapi.testclient import TestClient
        from scripts.inference.api_server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """健康检查端点。"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_models_endpoint(self, client):
        """模型列表端点。"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_chat_completions_no_model(self, client):
        """未加载模型时应返回 503。"""
        response = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}]
        })
        assert response.status_code == 503


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
