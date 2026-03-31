import streamlit as st
import requests

st.set_page_config(page_title="gAIa Doctor", page_icon="🌿")
st.title("🌿 gAIa: Diagnóstico Botânico")

uploaded_file = st.file_uploader("Carregue a foto da folha", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, width=300)
    
    if st.button("Analisar com IA"):
        with st.spinner("Processando..."):
            # Preparar o ficheiro para envio
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                # Chamada à tua API
                response = requests.post("https://gaia-api-sda8.onrender.com/predict", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Mostrar resultados básicos
                    st.success(f"Resultado: {data['prediction']} ({data['confidence']})")
                    
                    # Mostrar a análise (Gemini) de forma inteligente
                    st.subheader("📖 Relatório gAIa")
                    analysis = data.get("analysis", {})
                    
                    if isinstance(analysis, dict):
                        st.write(f"**Descrição:** {analysis.get('description', 'N/A')}")
                        
                        st.write("**Tratamento:**")
                        for t in analysis.get('treatment', []): st.write(f"- {t}")
                        
                        st.write("**Prevenção:**")  # <- adiciona aqui
                        for p in analysis.get('prevention', []): st.write(f"- {p}")
                    else:
                        st.write(analysis) # Se for apenas uma string
                else:
                    st.error(f"Erro no Servidor: {response.status_code}")
            except Exception as e:
                st.error(f"Erro de Conexão: Verifique se o FastAPI está ligado!")