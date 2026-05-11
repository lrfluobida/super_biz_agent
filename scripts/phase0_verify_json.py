"""Phase 0 验证 - Ollama qwen2.5:1.5b JSON 输出稳定性测试"""
import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:1.5b"

ROUTING_PROMPT = """你是一个查询意图分类器。将查询归为以下4类之一：
- direct: 简单明确，可直接检索
- decompose: 复合多主题，需拆分子查询
- step_back: 过于具体，需泛化抽象
- contextualize: 包含指代词，需补全上下文

输出严格JSON：{"intent": "...", "reason": "..."}"""

ROUTING_TESTS = [
    ("CPU使用率过高怎么排查", "direct"),
    ("CPU和内存的排查流程有什么区别", "decompose"),
    ("针对level=70的阈值设置建议", "step_back"),
    ("它为什么会触发", "contextualize"),
    ("磁盘使用率突然飙升怎么处理", "direct"),
    ("Prometheus告警规则和Grafana告警有什么区别", "decompose"),
    ("那个错误是什么原因", "contextualize"),
    ("scrape_interval设成15秒还是30秒好", "step_back"),
    ("Kubernetes Pod重启怎么排查", "direct"),
    ("日志采集和指标采集的配置差异", "decompose"),
    ("这个和之前的版本有什么变化", "contextualize"),
    ("针对timeout=30ms的配置优化建议", "step_back"),
    ("MySQL慢查询怎么定位", "direct"),
    ("内存泄漏和CPU飙升哪个更紧急", "decompose"),
    ("上面说的那个怎么查", "contextualize"),
]

DECOMPOSE_PROMPT = """将以下复杂问题拆分为2-4个独立的检索子问题。
规则：每个子问题聚焦单一主题，子问题之间无语义重叠，按重要性排序。
输出严格JSON：{"sub_queries": ["...", "..."], "original_topic": "..."}"""

DECOMPOSE_TESTS = [
    "CPU使用率高和内存泄漏的排查流程有什么区别",
    "Kubernetes Pod OOM和Node NotReady分别怎么排查",
    "Prometheus alertmanager和Grafana alerting的配置差异及适用场景",
    "日志采集使用Filebeat还是Fluentd，各有什么优劣",
    "数据库慢查询和连接池耗尽的排查方法对比",
]

STEPBACK_PROMPT = """将以下过于具体的问题泛化为通用检索查询。
规则：去除过于具体的数值/条件/特定名称，提取问题核心类别/领域，保持领域语义不变。
输出严格JSON：{"step_back_query": "...", "retained_specifics": "..."}"""

STEPBACK_TESTS = [
    "Prometheus的scrape_interval设成15秒好还是30秒好",
    "针对CPU使用率超过90%持续5分钟的告警阈值设置建议",
    "MySQL的max_connections设成500会不会太大",
    "Redis的内存淘汰策略选择allkeys-lru还是volatile-lru",
    "某个服务器在凌晨3点CPU突然飙升到95%怎么排查",
]

CONTEXTUALIZE_PROMPT = """你是一个查询改写工具。你的唯一任务是将对话消息转换为独立完整的检索句子。你只输出JSON，不说其他话。
对话历史：用户之前问了关于<<HISTORY_TOPIC>>的问题。
当前用户消息：<<CURRENT_QUERY>>
规则：将代词替换为具体实体，补全省略的主语/宾语。
输出JSON（只输出这个，不要解释）：{{"standalone_query": "补全后的完整检索句"}}"""

CONTEXTUALIZE_TESTS = [
    ("CPU使用率过高排查方法", "那内存呢"),
    ("Prometheus告警规则配置", "这个阈值怎么调整"),
    ("Pod一直处于Pending状态怎么办", "上面这个问题在生产环境怎么快速恢复"),
    ("MySQL慢查询优化方法", "那个参数具体设置多少合适"),
    ("Kubernetes节点故障处理", "它会不会影响已经运行的Pod"),
]


def call_ollama(messages, temperature=0.1):
    """调用 Ollama API"""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 512}
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    body = json.loads(resp.read())
    return body.get("message", {}).get("content", "")


def extract_json(text: str):
    """三层 JSON 提取策略"""
    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. 正则提取 {...}
    import re
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 3. 尝试 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None


def run_test(name, prompt_template, test_cases, build_message_fn, expected_keys):
    """运行一组测试"""
    print(f"\n{'='*60}")
    print(f"测试组: {name} ({len(test_cases)} 条)")
    print(f"{'='*60}")

    passed = 0
    total = len(test_cases)
    results = []

    for i, case in enumerate(test_cases):
        messages = build_message_fn(prompt_template, case)
        raw = ""
        try:
            raw = call_ollama(messages)
            parsed = extract_json(raw)

            if parsed is None:
                status = "FAIL_PARSE"
                detail = f"无法解析JSON, raw={raw[:150]}"
            else:
                missing = [k for k in expected_keys if k not in parsed]
                if missing:
                    status = "FAIL_MISSING_KEY"
                    detail = f"缺少字段: {missing}, parsed={parsed}"
                else:
                    status = "PASS"
                    detail = f"OK: {json.dumps(parsed, ensure_ascii=False)[:200]}"
                    passed += 1

        except Exception as e:
            status = "FAIL_ERROR"
            detail = f"异常: {e}, raw={raw[:150]}"

        results.append({"case": str(case), "status": status, "detail": detail})
        print(f"  [{i+1:2d}/{total}] {status}: {detail[:150]}")

    success_rate = passed / total * 100
    print(f"\n  >> {name} JSON解析成功率: {passed}/{total} = {success_rate:.1f}%")

    return passed, total, results


def build_routing_msg(prompt, case):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": case[0]},
    ]


def build_decompose_msg(prompt, case):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"用户问题：{case}"},
    ]


def build_stepback_msg(prompt, case):
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"用户问题：{case}"},
    ]


def build_contextualize_msg(prompt, case):
    history_topic, current_query = case
    filled_prompt = prompt.replace("<<HISTORY_TOPIC>>", history_topic).replace("<<CURRENT_QUERY>>", current_query)
    return [
        {"role": "system", "content": filled_prompt},
        {"role": "user", "content": current_query},
    ]


def main():
    print("=" * 60)
    print("Phase 0 验证 - Qwen2.5:1.5B JSON 输出稳定性")
    print(f"模型: {MODEL}")
    print("=" * 60)

    total_passed = 0
    total_cases = 0

    # 1. 路由分类测试
    p, t, _ = run_test(
        "路由分类 (Routing)", ROUTING_PROMPT, ROUTING_TESTS,
        build_routing_msg, ["intent", "reason"]
    )
    total_passed += p
    total_cases += t

    # 2. 任务分解测试
    p, t, _ = run_test(
        "任务分解 (Decompose)", DECOMPOSE_PROMPT, DECOMPOSE_TESTS,
        build_decompose_msg, ["sub_queries", "original_topic"]
    )
    total_passed += p
    total_cases += t

    # 3. Step-Back 泛化测试
    p, t, _ = run_test(
        "Step-Back 泛化", STEPBACK_PROMPT, STEPBACK_TESTS,
        build_stepback_msg, ["step_back_query", "retained_specifics"]
    )
    total_passed += p
    total_cases += t

    # 4. 上下文补全测试
    p, t, _ = run_test(
        "上下文补全 (Contextualize)", CONTEXTUALIZE_PROMPT, CONTEXTUALIZE_TESTS,
        build_contextualize_msg, ["standalone_query"]
    )
    total_passed += p
    total_cases += t

    overall_rate = total_passed / total_cases * 100
    print(f"\n{'='*60}")
    print(f"总成绩: {total_passed}/{total_cases} = {overall_rate:.1f}%")
    if overall_rate >= 95:
        print(">> PASS - Phase 0 验收通过! JSON 解析成功率 >= 95%")
    else:
        print(f">> FAIL - JSON 解析成功率 {overall_rate:.1f}% < 95% 阈值")
    print(f"{'='*60}")

    return 0 if overall_rate >= 95 else 1


if __name__ == "__main__":
    sys.exit(main())
