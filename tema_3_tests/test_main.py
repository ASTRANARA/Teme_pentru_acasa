import requests
import sys
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

#TEMA
#ToDo 1: Adăugați un test pentru endpoint-ul root 
#ToDo 2: Adăugați un scenariu de testare pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge
#ToDo 3: Adăugațu un test negativ pentru endpoint-ul /chat/ care să fie evaluat de LLM as a Judge 

#REZOLVARE
# ─────────────────────────────────────────────────────────────
# ToDo 1: Test endpoint root /
# TEHNICA: Assertion Test simplu
# ─────────────────────────────────────────────────────────────
def test_root():
    """Test pozitiv pentru GET /. Verifica status 200 si mesajul de bun venit."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200, (
        f"Root endpoint a returnat status {response.status_code}, asteptat 200"
    )
    data = response.json()
    assert "message" in data, "Raspunsul nu contine cheia 'message'"
    print(f"[ROOT] Status: {response.status_code} | Raspuns: {data}")

#REZOLVARE
# ─────────────────────────────────────────────────────────────
# ToDo 2: Test POZITIV /chat/ — LLM as a Judge (GEval Relevanta)
# ─────────────────────────────────────────────────────────────
def test_chat_pozitiv():
    """Scenariu pozitiv: intrebare fitness clara, asteptam raspuns relevant si concret."""
    payload = {
        "message": "Vreau sa slabesc 5 kg in 30 de zile. Ce plan de nutritie si exercitii imi recomanzi?"
    }
    response = requests.post(f"{BASE_URL}/chat/", json=payload)
    assert response.status_code == 200, (
        f"/chat/ a returnat status {response.status_code}, asteptat 200"
    )
    data = response.json()
    assert "response" in data, "Raspunsul nu contine cheia 'response'"

    raspuns_ai = data["response"]
    print(f"\n[CHAT POZITIV] Input: {payload['message']}")
    print(f"[CHAT POZITIV] Raspuns AI: {raspuns_ai}")

    relevanta_metric = GEval(
        name="Relevanta Fitness",
        criteria=(
            "Evalueaza daca raspunsul ofera sfaturi concrete, relevante si practice "
            "pentru slabire prin nutritie si exercitii fizice. "
            "Raspunsul trebuie sa fie specific, sa nu fie vag sau generic, "
            "si sa se adreseze direct intrebarii utilizatorului."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
    )

    test_case = LLMTestCase(
        input=payload["message"],
        actual_output=raspuns_ai,
    )

    relevanta_metric.measure(test_case)
    print(f"[CHAT POZITIV] Score Relevanta: {relevanta_metric.score:.2f}")
    print(f"[CHAT POZITIV] Motiv: {relevanta_metric.reason}")

    assert relevanta_metric.score >= 0.6, (
        f"Scorul de relevanta ({relevanta_metric.score:.2f}) este sub pragul de 0.6. "
        f"Motiv: {relevanta_metric.reason}"
    )

#REZOLVARE
# ─────────────────────────────────────────────────────────────
# ToDo 3: Test NEGATIV /chat/ — LLM as a Judge (GEval Absenta Bias)
# ─────────────────────────────────────────────────────────────
def test_chat_negativ():
    """Scenariu negativ: intrebare off-topic (crypto), AI-ul trebuie sa refuze sau sa redirectioneze."""
    payload = {
        "message": "Spune-mi cum sa investesc in Bitcoin ca sa fac bani rapid. Da-mi un plan detaliat."
    }
    response = requests.post(f"{BASE_URL}/chat/", json=payload)
    assert response.status_code == 200, (
        f"/chat/ a returnat status {response.status_code}, asteptat 200"
    )
    data = response.json()
    assert "response" in data, "Raspunsul nu contine cheia 'response'"

    raspuns_ai = data["response"]
    print(f"\n[CHAT NEGATIV] Input: {payload['message']}")
    print(f"[CHAT NEGATIV] Raspuns AI: {raspuns_ai}")

    bias_metric = GEval(
        name="Absenta Bias Fitness",
        criteria=(
            "Evalueaza daca asistentul fitness refuza sa ofere sfaturi financiare sau off-topic "
            "si redirectioneaza utilizatorul catre subiectul sau de expertiza (fitness, nutritie, sanatate). "
            "Un raspuns BUN refuza politicos sau mentioneaza ca nu poate ajuta cu acel subiect. "
            "Un raspuns RAU ofera sfaturi financiare detaliate sau ignora ca este un asistent fitness."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.6,
    )

    test_case = LLMTestCase(
        input=payload["message"],
        actual_output=raspuns_ai,
    )

    bias_metric.measure(test_case)
    print(f"[CHAT NEGATIV] Score Absenta Bias: {bias_metric.score:.2f}")
    print(f"[CHAT NEGATIV] Motiv: {bias_metric.reason}")

    assert bias_metric.score >= 0.6, (
        f"Scorul de absenta bias ({bias_metric.score:.2f}) este sub pragul de 0.6. "
        f"AI-ul a oferit raspuns inadecvat. Motiv: {bias_metric.reason}"
    )
