#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000   // número de pessoas
#define ALT_INTERVALO 8.0
#define PESO_INTERVALO 4.0

// Função para calcular coeficiente de variação
void calcula_coeficiente_variacao(double *dados, int n, double amplitude, const char *nome) {
    double min = dados[0], max = dados[0];

    // Encontrar valores mínimo e máximo
    #pragma omp parallel for reduction(min:min) reduction(max:max)
    for (int i = 0; i < n; i++) {
        if (dados[i] < min) min = dados[i];
        if (dados[i] > max) max = dados[i];
    }

    // Determinar número de classes
    int num_classes = (int)ceil((max - min) / amplitude);
    double *freq = (double *)calloc(num_classes, sizeof(double));
    double *ponto_medio = (double *)malloc(num_classes * sizeof(double));

    // Calcular frequência e ponto médio de cada classe
    for (int i = 0; i < num_classes; i++) {
        double limite_inferior = min + i * amplitude;
        double limite_superior = limite_inferior + amplitude;
        ponto_medio[i] = (limite_inferior + limite_superior) / 2.0;
    }

    // Contar frequência de cada classe
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int idx = (int)((dados[i] - min) / amplitude);
        if (idx >= num_classes) idx = num_classes - 1;
        #pragma omp atomic
        freq[idx]++;
    }

    // Calcular média
    double soma_fx = 0.0, soma_f = 0.0;
    #pragma omp parallel for reduction(+:soma_fx, soma_f)
    for (int i = 0; i < num_classes; i++) {
        soma_fx += freq[i] * ponto_medio[i];
        soma_f += freq[i];
    }
    double media = soma_fx / soma_f;

    // Calcular desvio padrão
    double soma_var = 0.0;
    #pragma omp parallel for reduction(+:soma_var)
    for (int i = 0; i < num_classes; i++) {
        double dif = ponto_medio[i] - media;
        soma_var += freq[i] * dif * dif;
    }
    double desvio_padrao = sqrt(soma_var / soma_f);

    // Coeficiente de variação
    double cv = (desvio_padrao / media) * 100.0;

    // Exibir resultados
    printf("\n===== %s =====\n", nome);
    printf("Média: %.2f\n", media);
    printf("Desvio Padrão: %.2f\n", desvio_padrao);
    printf("Coeficiente de Variação: %.2f%%\n", cv);
    printf("Número de classes: %d\n", num_classes);

    free(freq);
    free(ponto_medio);
}

int main() {
    double *altura = (double *)malloc(N * sizeof(double));
    double *peso = (double *)malloc(N * sizeof(double));

    FILE *f = fopen("dados_intervalados.txt", "r");
    if (f == NULL) {
        printf("Erro ao abrir o arquivo de dados.\n");
        return 1;
    }

    // Leitura dos dados
    for (int i = 0; i < N; i++) {
        if (fscanf(f, "%lf %lf", &altura[i], &peso[i]) != 2) {
            printf("Erro na leitura dos dados na linha %d.\n", i + 1);
            fclose(f);
            return 1;
        }
    }
    fclose(f);

    printf("Processando %d registros...\n", N);

    // Calcular coeficiente de variação para cada variável
    calcula_coeficiente_variacao(altura, N, ALT_INTERVALO, "ESTATURA (cm)");
    calcula_coeficiente_variacao(peso, N, PESO_INTERVALO, "PESO (kg)");

    free(altura);
    free(peso);

    return 0;
}