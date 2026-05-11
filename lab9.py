import os
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

HNSW_M               = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH       = 50
TOP_K_RETRIEVE       = 10
TOP_K_FINAL          = 3

BI_ENCODER_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CROSS_ENCODER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FRAGMENTOS_MEDICOS = [
    "Cefaleia pulsátil com fotofobia e fonofobia são os critérios diagnósticos primários da enxaqueca sem aura, conforme a Classificação Internacional das Cefaleias (ICHD-3). A dor tipicamente dura de 4 a 72 horas e é agravada por atividade física rotineira.",
    "O tratamento agudo da crise de enxaqueca inclui triptanos (sumatriptana 50-100mg VO) como primeira linha, AINEs (ibuprofeno 400-600mg) como alternativa e antieméticos (metoclopramida 10mg) para náusea associada. Ergotamínicos devem ser evitados em pacientes com risco cardiovascular.",
    "Hipertensão arterial sistêmica é definida como pressão arterial sistólica maior ou igual a 140 mmHg e/ou diastólica maior ou igual a 90 mmHg em pelo menos duas aferições em dias distintos. A hipertensão estágio 3 requer tratamento farmacológico imediato.",
    "O infarto agudo do miocárdio com supradesnivelamento do segmento ST (IAMCSST) requer reperfusão coronária imediata. A angioplastia primária (ICP) é preferível à fibrinólise quando disponível em até 120 minutos do primeiro contato médico.",
    "Diabetes mellitus tipo 2 é caracterizado por resistência à insulina e deficiência progressiva de secreção. O diagnóstico é estabelecido por glicemia de jejum maior ou igual a 126 mg/dL, hemoglobina glicada (HbA1c) maior ou igual a 6,5% ou glicemia 2h pós-sobrecarga maior ou igual a 200 mg/dL.",
    "A pneumonia adquirida na comunidade (PAC) é classificada pelo escore CURB-65. Escore 0-1: tratamento ambulatorial; 2: internação; 3 ou mais: UTI. Antibioticoterapia deve ser iniciada em até 4 horas do diagnóstico.",
    "Insuficiência cardíaca congestiva (ICC) com fração de ejeção reduzida tem como pilares terapêuticos: inibidores da ECA ou BRA, betabloqueadores (carvedilol, metoprolol, bisoprolol), antagonistas mineralocorticoides e inibidores SGLT2.",
    "O acidente vascular cerebral isquêmico (AVCi) requer neuroimagem imediata por TC sem contraste para excluir hemorragia. A trombólise com alteplase é indicada em até 4,5 horas do início dos sintomas na ausência de contraindicações.",
    "Asma brônquica é uma doença inflamatória crônica das vias aéreas com hiper-responsividade brônquica, obstrução reversível ao fluxo aéreo e sibilância expiratória. O controle é avaliado pelo ACQ e espirometria com broncodilatador.",
    "A sepse é definida como disfunção orgânica ameaçadora à vida causada por resposta desregulada do hospedeiro à infecção (Sepsis-3). O escore SOFA maior ou igual a 2 confirma disfunção orgânica. O bundle de 1 hora inclui hemoculturas, lactato sérico, antibióticos e fluidos.",
    "Hipotireoidismo primário resulta em TSH elevado e T4 livre reduzido. Sintomas incluem fadiga, ganho de peso, intolerância ao frio, constipação, bradicardia e pele ressecada. O tratamento de escolha é levotiroxina sódica com ajuste de dose pelo TSH.",
    "A insuficiência renal crônica (IRC) é definida por TFG menor que 60 mL/min por mais de 3 meses. As estadificações G1 a G5 baseiam-se na taxa de filtração glomerular. Estágio G5 indica necessidade de terapia renal substitutiva.",
    "Doença inflamatória intestinal compreende Doença de Crohn e Retocolite Ulcerativa. O diagnóstico é ileocolonoscópico com biópsia. A calprotectina fecal maior que 200 microgramas por grama é marcador sensível de inflamação intestinal ativa.",
    "A fibromialgia é uma síndrome de dor crônica difusa com hiperalgesia generalizada, fadiga e distúrbios do sono. Critérios diagnósticos ACR 2010 incluem índice de dor generalizada maior ou igual a 7 e escala de gravidade dos sintomas maior ou igual a 5.",
    "Lúpus eritematoso sistêmico (LES) é uma doença autoimune multissistêmica. Critérios diagnósticos SLICC incluem: artrite não erosiva, nefrite lúpica, anemia hemolítica, FAN positivo, anti-dsDNA, anti-Sm e complemento reduzido.",
    "A doença de Parkinson é uma desordem neurodegenerativa com tríade: tremor de repouso, rigidez plástica e bradicinesia. Resulta da perda de neurônios dopaminérgicos da substância negra. O tratamento de referência é levodopa associada à carbidopa.",
    "Epilepsia é definida por duas ou mais crises epilépticas não provocadas com intervalo maior que 24 horas. A classificação ILAE 2017 divide em: focal, generalizada e início desconhecido. O diagnóstico requer EEG e ressonância magnética de crânio.",
    "O transtorno depressivo maior requer pelo menos 5 sintomas do critério DSM-5 por duas semanas ou mais, incluindo humor deprimido ou anedonia. Primeira linha farmacológica são os ISRSs como sertralina e escitalopram.",
    "Artrite reumatoide é uma poliartrite inflamatória crônica simétrica de pequenas articulações com rigidez matinal superior a 60 minutos. Marcadores incluem fator reumatoide e anticorpo anti-CCP. Tratamento: metotrexato e agentes biológicos.",
    "A doença pulmonar obstrutiva crônica (DPOC) é definida por obstrução ao fluxo aéreo não totalmente reversível com VEF1/CVF menor que 0,70 pós-broncodilatador. Classificada pelos estadios GOLD 1 a 4. Tratamento inclui LAMA, LABA e corticoide inalatório.",
    "Osteoporose é caracterizada por densidade mineral óssea com T-score menor ou igual a menos 2,5 na densitometria. Fraturas por fragilidade no quadril, vértebra e rádio distal são a principal complicação. Tratamento inclui bifosfonatos e vitamina D.",
    "Hepatite C crônica é causada pelo vírus HCV. O diagnóstico é feito por anti-HCV e confirmado por HCV-RNA quantitativo. Os antivirais de ação direta pangenotípicos como sofosbuvir e velpatasvir alcançam taxas de cura superiores a 95 por cento em 12 semanas.",
]


def carregar_bi_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        modelo = SentenceTransformer(BI_ENCODER_MODEL)
        print(f"  bi-encoder carregado: {BI_ENCODER_MODEL}")
        return modelo, "sentence_transformer"
    except Exception:
        print("  bi-encoder HuggingFace indisponivel, usando TF-IDF local como fallback")
        vetorizador = TfidfVectorizer(max_features=512)
        vetorizador.fit(FRAGMENTOS_MEDICOS)
        return vetorizador, "tfidf"


def gerar_embedding(modelo, tipo, textos):
    if tipo == "sentence_transformer":
        vecs = modelo.encode(textos, normalize_embeddings=True)
    else:
        vecs = modelo.transform(textos).toarray().astype(np.float32)
        vecs = normalize(vecs)
    return np.array(vecs, dtype=np.float32)


print("=" * 60)
print("PASSO 1 — Construção e Indexação do Grafo HNSW")
print("=" * 60)

print("\ncarregando modelo de embedding...")
bi_encoder, encoder_tipo = carregar_bi_encoder()

print(f"gerando embeddings para {len(FRAGMENTOS_MEDICOS)} fragmentos...")
embeddings = gerar_embedding(bi_encoder, encoder_tipo, FRAGMENTOS_MEDICOS)
DIM = embeddings.shape[1]
print(f"dimensao dos vetores : {DIM}")

indice_hnsw = faiss.IndexHNSWFlat(DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
indice_hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
indice_hnsw.hnsw.efSearch        = HNSW_EF_SEARCH
indice_hnsw.add(embeddings)

print(f"indice HNSW construido:")
print(f"  M (conexoes/camada)  : {HNSW_M}")
print(f"  ef_construction      : {HNSW_EF_CONSTRUCTION}")
print(f"  ef_search            : {HNSW_EF_SEARCH}")
print(f"  total de vetores     : {indice_hnsw.ntotal}")


def hyde_transform(query_coloquial):
    template = (
        "Voce e um medico especialista. Com base na queixa do paciente abaixo, "
        "escreva um paragrafo tecnico de manual medico usando terminologia clinica formal.\n\n"
        f"Queixa: {query_coloquial}\n\nDocumento hipotetico:"
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": template}],
            max_tokens=300,
            temperature=0.3,
        )
        doc = resp.choices[0].message.content.strip()
        print("  [HyDE] gerado via OpenAI")
    except Exception:
        mapa = {
            "cabeca":    "Cefaleia pulsatil com fotofobia e fonofobia compativel com enxaqueca sem aura. Dor unilateral de moderada a intensa agravada por atividade fisica.",
            "coracao":   "Taquicardia com dispneia aos esforcos e ortopneia sugestivo de disfuncao ventricular esquerda com insuficiencia cardiaca congestiva.",
            "acucar":    "Hiperglicemia em jejum compativel com diabetes mellitus tipo 2 descompensado com poliuria, polidipsia e polifagia.",
            "juntas":    "Poliartrite inflamatoria simetrica de pequenas articulacoes com rigidez matinal superior a 60 minutos sugestivo de artrite reumatoide.",
            "tosse":     "Tosse produtiva cronica com sibilancia expiratoria e dispneia compativel com DPOC ou asma brônquica. Espirometria indicada.",
        }
        doc = next(
            (v for k, v in mapa.items() if k in query_coloquial.lower()),
            f"Quadro clinico descrito como '{query_coloquial}'. Avaliacao medica com exames complementares indicada.",
        )
        print("  [HyDE] gerado via fallback local")

    vetor = gerar_embedding(bi_encoder, encoder_tipo, [doc])
    return doc, vetor


def buscar_hnsw(vetor_query, k=TOP_K_RETRIEVE):
    scores, indices = indice_hnsw.search(vetor_query, k)
    return [
        {"indice": int(idx), "score": float(sc), "texto": FRAGMENTOS_MEDICOS[idx]}
        for sc, idx in zip(scores[0], indices[0]) if idx >= 0
    ]


def reranking_cross_encoder(query_original, candidatos, top_k=TOP_K_FINAL):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CROSS_ENCODER_MODEL)
        scores_ce = ce.predict([[query_original, c["texto"]] for c in candidatos])
        for i, c in enumerate(candidatos):
            c["score_cross_encoder"] = float(scores_ce[i])
        print("  [Cross-Encoder] re-ranking via modelo neural")
    except Exception:
        print("  [Cross-Encoder] indisponivel, usando score HNSW como fallback")
        for c in candidatos:
            c["score_cross_encoder"] = c["score"]
    return sorted(candidatos, key=lambda x: x["score_cross_encoder"], reverse=True)[:top_k]


def pipeline_rag(query_coloquial):
    print("\n" + "=" * 60)
    print(f"QUERY: {query_coloquial}")
    print("=" * 60)

    print("\n--- PASSO 2: HyDE ---")
    doc_hyde, vetor_hyde = hyde_transform(query_coloquial)
    print(f"  doc: \"{doc_hyde[:150]}...\"")

    print(f"\n--- PASSO 3: Busca HNSW — Top-{TOP_K_RETRIEVE} ---")
    candidatos = buscar_hnsw(vetor_hyde)
    for i, c in enumerate(candidatos, 1):
        print(f"  [{i:02d}] score={c['score']:.4f} | {c['texto'][:70]}...")

    print(f"\n--- PASSO 4: Re-ranking Cross-Encoder — Top-{TOP_K_FINAL} ---")
    finais = reranking_cross_encoder(query_coloquial, candidatos)
    print("\n  documentos finais para injecao no LLM:")
    for i, doc in enumerate(finais, 1):
        print(f"\n  [{i}] score_ce={doc['score_cross_encoder']:.4f}")
        print(f"       {doc['texto']}")

    return finais


if __name__ == "__main__":
    queries_teste = [
        "dor de cabeça latejante e luz incomodando",
        "coracao acelerado e falta de ar quando subo escada",
        "acucar alto no sangue e muita sede",
        "dor nas juntas das maos que piora de manha",
        "tosse que nao passa e chiado no peito",
    ]
    for query in queries_teste:
        pipeline_rag(query)
        print()