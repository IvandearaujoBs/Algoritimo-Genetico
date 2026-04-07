import numpy as np

def main():
    total_nfe = 0  
    SRFinal = 0
    numRepeticoes = 100
    
    # Ótimo real da função BF1
    f_otimo_real = 0  # Mínimo global em (0,0) vale 0
    
    for cont in range(numRepeticoes):
        # --- Parâmetros para BF1 ---
        nvars = 2  # Quantas variáveis o problema tem
        lb, ub = -50, 50  # Domínio da função BF1: [-50, 50]
        SizePop = 100  # população
        NumGeracoes = 0
        NumGeracoesMax = 1000  
        NFE = 0  # Número de avaliações de fitness
        
        # Inicialização da população
        PopIni = np.zeros((SizePop, nvars + 1))
        for i in range(SizePop):
            PopIni[i, :nvars] = lb + (ub - lb) * np.random.rand(nvars)
            x1, x2 = PopIni[i, 0], PopIni[i, 1]
            # Função BF1 (Bohachevsky I)
            PopIni[i, 2] = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7
            NFE += 1
        
        PopAtual = PopIni.copy()
        MinFitAtual = np.min(PopAtual[:, 2])
        
        # Guarda o melhor indivíduo (elitismo)
        melhor_idx = np.argmin(PopAtual[:, 2])
        melhor_individuo = PopAtual[melhor_idx].copy()
        
        ContFitRep = 0 
        
        print(f"\n>>> Execução {cont + 1}/100")
        
        while (ContFitRep < 10) and (NumGeracoes < NumGeracoesMax):
            MinFitAnterior = MinFitAtual
            
            # Torneio com tamanho 3
            Filhos = np.zeros((SizePop, nvars + 1))
            
            for i in range(SizePop - 1):  # -1 para o elitista
                # Torneio tamanho 3 para pai 1
                idx1 = np.random.randint(SizePop, size=3)
                p1 = PopAtual[idx1[np.argmin(PopAtual[idx1, 2])]]
                
                # Torneio tamanho 3 para pai 2
                idx2 = np.random.randint(SizePop, size=3)
                p2 = PopAtual[idx2[np.argmin(PopAtual[idx2, 2])]]
                
                # Crossover BLX-alpha
                if np.random.rand() < 0.9:
                    alpha = 0.5
                    for j in range(nvars):
                        d = abs(p1[j] - p2[j])
                        min_val = min(p1[j], p2[j])
                        max_val = max(p1[j], p2[j])
                        Filhos[i, j] = np.random.uniform(min_val - alpha*d, max_val + alpha*d)
                        Filhos[i, j] = np.clip(Filhos[i, j], lb, ub)
                else:
                    Filhos[i, :2] = p1[:2]
            
            # Elitismo: melhor indivíduo vai para próxima geração
            Filhos[-1, :2] = melhor_individuo[:2]
            
            # Mutação Gaussiana com taxa adaptativa
            taxa_mutacao = 0.15 * (1 - NumGeracoes/NumGeracoesMax) + 0.05
            for i in range(SizePop - 1):
                if np.random.rand() < taxa_mutacao:
                    for j in range(nvars):
                        if np.random.rand() < 0.3:
                            passo = 0.5 * (1 - NumGeracoes/NumGeracoesMax) + 0.1
                            Filhos[i, j] += np.random.normal(0, passo)
                            Filhos[i, j] = np.clip(Filhos[i, j], lb, ub)
            
            # Avaliação dos Filhos (Função BF1)
            for i in range(SizePop):
                x1, x2 = Filhos[i, 0], Filhos[i, 1]
                Filhos[i, 2] = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1) - 0.4*np.cos(4*np.pi*x2) + 0.7
                NFE += 1
            
            PopAtual = Filhos.copy()
            NumGeracoes += 1
            MinFitAtual = np.min(PopAtual[:, 2])
            
            # Atualiza melhor indivíduo
            melhor_idx = np.argmin(PopAtual[:, 2])
            if PopAtual[melhor_idx, 2] < melhor_individuo[2]:
                melhor_individuo = PopAtual[melhor_idx].copy()
            
            # Critério de parada
            if abs(MinFitAtual - MinFitAnterior) < 1e-10:
                ContFitRep += 1
            else:
                ContFitRep = 0
        
        total_nfe += NFE
        
        # Critério de sucesso para BF1 (tolerância 0.01)
        if abs(MinFitAtual - f_otimo_real) < 0.01:
            SRFinal += 1
            print(f"  [OK] Sucesso! Fit: {MinFitAtual:.8f} | NFE: {NFE}")
            print(f"       x1 = {melhor_individuo[0]:.6f}, x2 = {melhor_individuo[1]:.6f}")
        else:
            print(f"  [!] Falha. Fit: {MinFitAtual:.8f} | NFE: {NFE}")
            print(f"       x1 = {melhor_individuo[0]:.6f}, x2 = {melhor_individuo[1]:.6f}")

    print("\n" + "="*40)
    print("RESULTADOS FINAIS - FUNÇÃO BF1")
    print(f"NFE Médio: {total_nfe / numRepeticoes:.0f}")
    print(f"Taxa de Sucesso (SR): {SRFinal}%")
    print("="*40)

if __name__ == "__main__":
    main()