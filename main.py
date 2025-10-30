import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy import special, optimize
import scipy.stats as stats

FIXED_BINOMIAL_Nt = 1000 # como a dataset fornece a fracao da perda de pacotes, sao assumidos FIXED_BINOMIAL_Nt pacotes transmitidos


def main():
    original_df = pd.read_csv('ndt_tests_corrigido.csv')

    # Analise Exploratoria de data

    # Calcule as estatísticas descritivas: média, mediana, variância, desvio padrão
    # calc_descriptive_statistics(original_df)

    # Para os clientes e servidores selecionados, crie os gráficos: histograma, boxplot, scatter plot
    # create_descriptive_statistics_grafs(original_df)

    # MLE
    mle_results = calculate_mle(original_df)

    # Inferencia Bayesiana
    calculate_bayesian_inference(original_df, mle_results)


def calc_descriptive_statistics(original_df):
    print(original_df.describe())

    for var in ["client", "server"]:
        clients_servers = original_df[var].unique()
        for cs in clients_servers:
            print(cs)
            df = original_df[original_df[var] == cs]
            statistics = df.describe()
            print(statistics)
            medians = df.median(numeric_only=True)
            vars = df.var(numeric_only=True)
            print(f"Medianas:\n{medians}\nVariancias:\n{vars}")
            columns = [ column for column in list(df.columns) if column not in ['client', 'server', 'timestamp'] ]
            for key in columns:
                quantils = df[key].quantile([0.9, 0.95, 0.99])
                print(f"{quantils}\n\n")

def create_descriptive_statistics_grafs(original_df, clients_select = ["client01", "client10"]):
    variaveis = {
        "download_throughput_bps": "Throughput Download (bps)",
        "upload_throughput_bps": "Throughput Upload (bps)",
        "rtt_download_sec": "RTT Download (s)",
        "rtt_upload_sec": "RTT Upload (s)",
        "packet_loss_percent": "Perda de Pacotes (%)"
    }
    for client in clients_select:
        df = original_df[original_df["client"] == client]
        for var, label in variaveis.items():
            data = df[var]
            # HISTOGRAMA
            plt.figure(figsize=(6,4))
            plt.hist(data, bins=100, color="skyblue", edgecolor="black")
            plt.title(f"Histograma {client} - {label}")
            plt.xlabel(label)
            plt.ylabel("Frequência")
            plt.tight_layout()
            plt.savefig(f"figuras/hist_{var}_{client}.png", dpi=300)
            plt.close()

            # BOXPLOT
            plt.figure(figsize=(4,5))
            plt.boxplot(data, vert=True)
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


def calculate_mle(original_df, clients_select = ["client01", "client10"]):
    gamma_funcs = (gamma_mle, plot_mle_gamma)
    normal_funcs = (normal_mle, plot_mle_normal)
    binomial_funcs = (binomial_mle, plot_mle_binomial)
    variaveis = {
        "download_throughput_bps": gamma_funcs,
        "upload_throughput_bps": gamma_funcs,
        "rtt_download_sec": normal_funcs,
        "rtt_upload_sec": normal_funcs,
        "packet_loss_percent": binomial_funcs
    }
    results = {}
    for client in clients_select:
        client_result = {}
        df = original_df[original_df["client"] == client]
        for var, funcs in variaveis.items():
            print(f"MLE {client} - {var}")
            data = df[var].to_numpy()

            mle_calc = funcs[0]
            mle_result = mle_calc(data)
            client_result[var] = mle_result

            mle_plot = funcs[1]
            mle_plot(data, client, var, mle_result)

        results[client] = client_result

    return results

def gamma_mle(data: np.ndarray):
    y = data

    mean_y = np.mean(y)
    mean_log_y = np.mean(np.log(y))

    # Valor alvo para encontrar k
    s = np.log(mean_y) - mean_log_y

    # Função cuja raiz é k_mle
    log_like_k_derivative = lambda k: np.log(k) - special.digamma(k) - s

    # Estimativa inicial (método dos momentos)
    k0 = mean_y**2 / y.var(ddof=1)

    # Resolver numericamente para k_mle
    k_mle = optimize.newton(log_like_k_derivative, k0)

    # Estimar beta (rate)
    beta_mle = k_mle / mean_y

    print(f"GAMMA MLE k^={k_mle} B^={beta_mle}")
    return k_mle, beta_mle


def normal_mle(data: np.ndarray):
    meu_mle = normal_mu_mle(data)
    var_mle = normal_var_mle(data, meu_mle)

    print(f"NORMAL MLE u^={meu_mle} o2^={var_mle}")
    return meu_mle, var_mle

def normal_mu_mle(data: np.ndarray):
    return np.mean(data)

def normal_var_mle(data, meu_mle: np.ndarray):
    return np.mean((data - meu_mle) ** 2)

def binomial_mle(data: np.ndarray):
    p_hat = np.mean((data / 100) * FIXED_BINOMIAL_Nt)

    print(f"BINOMIAL MLE p^={p_hat}")
    return p_hat


def plot_mle_gamma(data, client, var, params):
    k_mle = params[0]
    theta = 1 / params[1]

    # Histograma
    plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Função densidade do modelo
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.gamma.pdf(x, a=k_mle, scale=theta), 'r-', lw=2, label='Gamma MLE')
    plt.xlabel('data')
    plt.ylabel('Densidade')
    plt.title(f'Histograma + PDF ajustada ({var}, {client})')
    plt.legend()
    plt.savefig(f"figuras/MLE_hist_pdf_gamma_{var}_{client}.png", dpi=300)

    # 2️⃣ QQ plot
    stats.probplot(data, dist=stats.gamma, sparams=(k_mle, 0, theta), plot=plt)
    plt.title(f'QQ Plot dos data vs Gamma ajustada ({var}, {client})')
    plt.savefig(f"figuras/MLE_qqplot_mle_gamma_{var}_{client}.png", dpi=300)
    plt.close()


def plot_mle_normal(data, client, var, params):
    meu_mle = params[0]
    std_mle = np.sqrt(params[1])

    # Histograma
    plt.hist(data, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')

    # Função densidade do modelo
    x = np.linspace(min(data), max(data), 100)
    plt.plot(x, stats.norm.pdf(x, loc=meu_mle, scale=std_mle), 'r-', lw=2, label='Normal MLE')
    plt.xlabel('data')
    plt.ylabel('Densidade')
    plt.title(f'Histograma + PDF ajustada ({var}, {client})')
    plt.legend()
    plt.savefig(f"figuras/MLE_hist_pdf_normal_{var}_{client}.png", dpi=300)
    plt.close()

    # 2️⃣ QQ plot
    stats.probplot(data, dist=stats.norm, sparams=(meu_mle, std_mle), plot=plt)
    plt.title(f'QQ Plot dos data vs Normal ajustada ({var}, {client})')
    plt.savefig(f"figuras/MLE_qqplot_mle_normal_{var}_{client}.png", dpi=300)
    plt.close()

def plot_mle_binomial(data, client, var, params):
    p_mle = params / 100

    # Converter percentuais em contagem de sucessos
    data_num_packet_loss = np.round((data / 100) * FIXED_BINOMIAL_Nt).astype(int)

    # PMF teórica
    x = np.arange(0, FIXED_BINOMIAL_Nt+1)
    pmf_binom = stats.binom.pmf(x, FIXED_BINOMIAL_Nt, p_mle)

    # Plot
    plt.figure(figsize=(10,5))
    plt.hist(data_num_packet_loss, bins="auto", density=True, alpha=0.6, color='skyblue', edgecolor='black', label='data reais')
    plt.plot(x, pmf_binom, 'ro-', lw=2, label=f'Binomial MLE (p={p_mle:.3f})')
    plt.xlabel('Perda (%) convertida em contagem')
    plt.ylabel('Probabilidade')
    plt.title(f'Histograma + PMF Binomial ({var}, {client})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figuras/MLE_hist_pmf_binomial_{var}_{client}.png", dpi=300)
    plt.close()


def calculate_bayesian_inference(original_df, mle_results, clients_select = ["client01", "client10"]):
    gamma_funcs = (posterior_gamma, posterior_gamma_prediction)
    normal_funcs = (posterior_normal, posterior_normal_prediction)
    binomial_funcs = (posterior_beta, posterior_beta_prediction)
    variaveis = {
        "download_throughput_bps": gamma_funcs,
        "upload_throughput_bps": gamma_funcs,
        "rtt_download_sec": normal_funcs,
        "rtt_upload_sec": normal_funcs,
        "packet_loss_percent": binomial_funcs
    }
    results = {}
    for client in clients_select:
        client_result = {}
        df = original_df[original_df["client"] == client]
        for var, funcs in variaveis.items():
            print(f"Bayesian Inference {client} - {var}")
            bayesian_results = {}

            data = df[var].to_numpy()
            idxs = np.arange(len(data))
            np.random.shuffle(idxs)
            train_split = int(0.7 * len(data))
            train_data = data[idxs[:train_split]]
            test_data = data[idxs[:train_split]]

            calculate_posterior_params = funcs[0]
            calculate_posterior_predictions = funcs[1]

            like_params = mle_results[client][var]
            posterior_params = calculate_posterior_params(train_data, like_params)
            bayesian_results["posterior_params"] = posterior_params

            bayesian_results["test_mean"] = np.mean(test_data)
            bayesian_results["test_var"] = np.var(test_data)
            pred_mean, pred_var = calculate_posterior_predictions(posterior_params, like_params, test_data)
            bayesian_results["pred_mean"] = pred_mean
            bayesian_results["pred_var"] = pred_var

            print(bayesian_results)
            client_result[var] = bayesian_results
        
        results[client] = client_result
    
    return results


def posterior_normal(data, like_params, prior_mu = 0, prior_var = 10 ** 6):
    n = len(data)
    data_mu = np.mean(data)
    like_var =like_params[1]
    post_var = posterior_var_normal(prior_var, like_var, n)
    post_mu = posterior_mu_normal(post_var, prior_mu, prior_var, like_var, data_mu, n)
    return post_mu, post_var

def posterior_var_normal(prior_var, like_var, n):
    return ((1 / prior_var) + (n / like_var)) ** (-1)

def posterior_mu_normal(post_var, prior_mu, prior_var, like_var, data_mu, n):
    return post_var * ((prior_mu / prior_var) + (n * data_mu / like_var))

def posterior_beta(data, like_params, prior_a = 1, prior_b = 1):
    data_num_packet_loss = data * FIXED_BINOMIAL_Nt
    x_tot = np.sum(data_num_packet_loss)
    n_tot = FIXED_BINOMIAL_Nt * len(data_num_packet_loss)
    post_a = prior_a + x_tot
    post_b = prior_b + (n_tot - x_tot)
    return post_a, post_b

def posterior_gamma(data, like_params, prior_a = 10 ** -3, prior_b = 10 ** -3):
    like_k = like_params[0]
    n = len(data)
    post_a = prior_a + n * like_k
    post_b = prior_b + np.sum(data)
    return post_a, post_b

def posterior_normal_prediction(post_params, like_params, data):
    return post_params

def posterior_beta_prediction(post_params, like_params, data):
    # data nao foi dividido por 100, pois no dataset de teste os percentuais se encontram no intervalo de [0, 100]
    # assim caso a predicao fosse feita utilizando intervalo de dados de [0, 1], a media e var previstas seriam impactadas pela mesma proporcao
    n_tot = FIXED_BINOMIAL_Nt * len(data) 
    post_a = post_params[0]
    post_b = post_params[1]
    pred_mean = post_a / (post_a + post_b)
    pred_var = post_a * post_b * (post_a + post_b + n_tot) / (((post_a + post_b) ** 2) * (post_a + post_b + 1) * n_tot)

    return pred_mean, pred_var

def posterior_gamma_prediction(post_params, like_params, data):
    k = like_params[0]
    post_a = post_params[0]
    post_b = post_params[1]
    pred_mean = None
    pred_var = None
    if post_a > 1:
        pred_mean = k * post_b / (post_a - 1)
    if post_a > 2:
        pred_var = k * (k * post_a - 1) * (post_b ** 2) / (((post_a - 1) ** 2) * (post_a - 2))
    return pred_mean, pred_var


main()