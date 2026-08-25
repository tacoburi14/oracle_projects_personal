"""CIU 채점용 LLM 프롬프트 템플릿. judges.py(Anthropic/OpenAI 스텁)와
langgraph_judge.py(Ollama/Qwen 실제 경로)가 공용으로 쓴다."""

PROMPT_TEMPLATE = """당신은 실어증 담화 분석 전문가입니다. 아래 5단계 절차에 따라 발화를 CIU(Correct Information Unit) 기준으로 채점하세요.

## 절차
1. 형태 분석: 각 어절을 어간과 형태소(조사/어미)로 분해한다.
2. 어휘의미 분석: 어간의 의미 범주를 판단한다 (person/object/action/location/aspect/modality/bound_noun 중 하나, 해당 없으면 null).
3. 의미역 분석: 문장 내 의미역을 부여한다 (Agent/Theme/Predicate/Location/Aspect/Modality 중 하나, 해당 없으면 null).
4. 맥락적합성 평가: [장면 정보]와 대조하여 accurate(정확)/relevant(관련)/not_redundant(비중복)를 판정한다.
   - 이미 확립된 의미역(예: Theme)을 다른 표현으로 재지칭할 뿐 새 정보가 없으면 not_redundant=false.
5. 필러(어, 음, 그 등)는 disfluency="filler", 자기수정으로 폐기된 조각(예: "남자"→"남자아이가")은
   disfluency="false_start"로 표시하고 accurate/relevant/not_redundant는 모두 false로 둔다.
   정상적으로 채점 대상인 어절은 disfluency를 null로 둔다.

## 출력 형식
다른 설명 없이 아래 JSON 스키마와 정확히 일치하는 JSON만 출력하라.
{{
  "tokens": [
    {{"surface": "어절 원문", "disfluency": null,
      "category": null, "role": null,
      "accurate": true, "relevant": true, "not_redundant": true,
      "note": "판정 근거 한 줄"}}
  ]
}}

## 예시 (판단 기준을 보여주는 참고용 — 실제 채점 대상과는 다른 장면/발화)

장면 정보: 남자아이 한 명이 하늘에서 연을 날리고 있다.

발화: 어 남자 남자아이가 그 어 연울 사용 해서 하늘 앞에서 음 날리고 있는 것 같 은데

기대 출력:
{{
  "tokens": [
    {{"surface": "어", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"}},
    {{"surface": "남자", "disfluency": "false_start", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "바로 다음 어절 '남자아이가'로 자기수정됨"}},
    {{"surface": "남자아이가", "disfluency": null, "category": "person", "role": "Agent",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "그림 속 인물(Agent)과 일치"}},
    {{"surface": "그", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"}},
    {{"surface": "어", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"}},
    {{"surface": "연울", "disfluency": null, "category": "object", "role": "Theme",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "발음 오류('연을'의 변이형)지만 문맥상 명료, 그림 속 대상과 일치"}},
    {{"surface": "사용해서", "disfluency": null, "category": "action", "role": "Predicate",
      "accurate": true, "relevant": true, "not_redundant": false, "note": "이미 확립된 Theme('연을')의 재지칭 — 새 정보 없이 의미 중복"}},
    {{"surface": "하늘", "disfluency": null, "category": "location", "role": "Location",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "배경(Location)과 일치"}},
    {{"surface": "앞에서", "disfluency": null, "category": "location", "role": "Location",
      "accurate": false, "relevant": true, "not_redundant": true, "note": "'하늘 앞에서'는 성립하지 않는 위치 관계 — 정확성 실패"}},
    {{"surface": "음", "disfluency": "filler", "category": null, "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "채움말"}},
    {{"surface": "날리고", "disfluency": null, "category": "action", "role": "Predicate",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "그림 속 동작(Predicate)과 일치"}},
    {{"surface": "있는", "disfluency": null, "category": "aspect", "role": "Aspect",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "진행상 표현 — 관련성 있음, 중복 아님"}},
    {{"surface": "것", "disfluency": null, "category": "bound_noun", "role": null,
      "accurate": false, "relevant": false, "not_redundant": true, "note": "지시 대상 없는 형식명사"}},
    {{"surface": "같은데", "disfluency": null, "category": "modality", "role": "Modality",
      "accurate": true, "relevant": true, "not_redundant": true, "note": "화자의 불확실성 표현 — 과제 수행과 관련"}}
  ]
}}

이 예시처럼: 필러/자기수정은 accurate·relevant를 false로, 이미 나온 정보의 재지칭은 not_redundant를 false로,
그림과 안 맞는 표현(위치 관계 등)은 accurate를 false로 판정하라.

## 실제 채점 대상

장면 정보: {scene_description}

발화 (전사문 — STT로 변환된 원문, 필러/멈춤 표시가 있을 수도 없을 수도 있음): {transcript}

위 예시와 같은 판단 기준으로, 위 [출력 형식]의 JSON 스키마에 맞춰 이 발화를 채점하라.
"""
