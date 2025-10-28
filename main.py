import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

original_df = pd.read_csv('ndt_tests_corrigido.csv')

# Analise Exploratoria de Dados

# Calcule as estatísticas descritivas: média, mediana, variância, desvio padrão
print(original_df.describe())

var = "client"
# var = "server"
# clients_servers = original_df[var].unique()
# for cs in clients_servers:
#     print(cs)
#     df = original_df[original_df[var] == cs]
#     estatisticas = df.describe()
#     print(estatisticas)
#     medianas = df.median(numeric_only=True)
#     variancias = df.var(numeric_only=True)
#     print(f"Medianas:\n{medianas}\nVariancias:\n{variancias}")
#     columns = [ column for column in list(df.columns) if column not in ['client', 'server', 'timestamp'] ]
#     for key in columns:
#         quantils = df[key].quantile([0.9, 0.95, 0.99])
#         print(f"{quantils}\n\n")

# Para os clientes e servidores selecionados, crie os gráficos: histograma, boxplot, scatter plot

variaveis = {
    "download_throughput_bps": "Throughput Download (bps)",
    "upload_throughput_bps": "Throughput Upload (bps)",
    "rtt_download_sec": "RTT Download (s)",
    "rtt_upload_sec": "RTT Upload (s)",
    "packet_loss_percent": "Perda de Pacotes (%)"
}
clients_select = ["all"]
# clients_select = ["client01", "client10"]
for client in clients_select:
    # df = original_df[original_df["client"] == client]
    df = original_df
    for var, label in variaveis.items():
        dados = df[var].dropna()
         # HISTOGRAMA
        plt.figure(figsize=(6,4))
        plt.hist(dados, bins=100, color="skyblue", edgecolor="black")
        plt.title(f"Histograma {client} - {label}")
        plt.xlabel(label)
        plt.ylabel("Frequência")
        plt.tight_layout()
        plt.savefig(f"figuras/hist_{var}_{client}.png", dpi=300)
        plt.close()

        # BOXPLOT
        plt.figure(figsize=(4,5))
        plt.boxplot(dados, vert=True)
        plt.title(f"Boxplot {client} - {label}")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(f"figuras/box_{var}_{client}.png", dpi=300)
        plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(df["packet_loss_percent"], df["download_throughput_bps"], alpha=0.7, color="royalblue")
    plt.title("Relação entre Perda de Pacotes e Throughput de Download")
    plt.xlabel("Perda de Pacotes (%)")
    plt.ylabel("Throughput Download (bps)")
    plt.grid(True)
    plt.savefig(f"figuras/packet_loss_x_download_throughput_{client}.png", dpi=300)
