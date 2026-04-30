import numpy as np
from scipy.optimize import root
from scipy.special import digamma
import scipy.special as sp

def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log

def optimize_q_s(alpha, C, *messages):
    """
    Berechnet die optimale variationelle Verteilung q(s) über das Newton-Verfahren,
    wobei die Likelihood durch eine Dirichlet-Verteilung parametrisiert ist.
    
    Inputs:
    - alpha:  numpy array (16, 8) - Dirichlet-Parameter-Matrix für p(o | s)
    - C:      numpy array (16,)   - Ziel-Prior p~(o)
    - messages: beliebige Anzahl an 1D-Arrays (Größe 8) für eingehende Nachrichten
    
    Returns:
    - q_s_star: numpy array (8,)   - Die konvergierte, optimale Verteilung
    """
    
    # --- 1. Vorberechnung der Dirichlet-Erwartungswerte ---
    
    # Summe der Alphas über die Beobachtungen (Spaltensummen, shape: (8,))
    alpha_sum = np.sum(alpha, axis=0)
    
    # Erwartungswert der Likelihood-Matrix E[A] (shape: (16, 8))
    # Das entspricht dem Term alpha_{i,s} / sum_k(alpha_{k,s})
    E_A = alpha / alpha_sum
    
    # Erwartungswert von A * log(A) unter der Dirichlet-Verteilung
    # E_q(A)[A_is * log(A_is)] = E[A_is] * (psi(alpha_is + 1) - psi(alpha_sum + 1))
    E_A_log_A = E_A * (digamma(alpha + 1) - digamma(alpha_sum + 1))
    
    # Wir summieren direkt über die Spalten (axis=0), da dies der linke Term 
    # in der untersten Zeile deines zweiten Bildes ist. (shape: (8,))
    expected_A_log_A_sum = np.sum(E_A_log_A, axis=0)
    
    # Eingehende Nachrichten im Log-Raum addieren (statisch, daher außerhalb der Schleife)
    log_messages = sum(safelog(msg) for msg in messages)
    
    # --- 2. Verschachtelte Hilfsfunktionen für den Solver ---
    
    def softmax(x):
        """Numerisch stabiler Softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def error_function(q_s):
        """Berechnet das Residuum F(q_s) für den Solver"""
        
        # Vorhersage der Beobachtung (q_o) basierend auf der erwarteten Likelihood
        q_o = E_A @ q_s
        
        # Fehler-Term rho berechnen (entspricht exakt dem Exponenten aus dem 2. Bild)
        # Term 1: expected_A_log_A_sum (ersetzt das alte -h_p_A)
        # Term 2: E_A.T @ (log(C) - log(q_o))
        rho = expected_A_log_A_sum + E_A.T @ (safelog(C) - safelog(q_o))
        
        # Rückgabe der Differenz (soll 0 werden)
        return q_s - softmax(rho + log_messages)

    # --- 3. Vorbereitung und Optimierung ---
    
    num_states = alpha.shape[1]
    
    # Startwert für den Solver (Gleichverteilung)
    q_s_init = np.ones(num_states) / num_states
    
    # Solver aufrufen
    solution = root(error_function, x0=q_s_init, method='hybr')
    
    # --- 4. Ergebnisverarbeitung ---
    
    if solution.success:
        q_s_star = solution.x
        
        # Numerische Sicherheitsmaßnahme: Werte zwischen 0 und 1 clippen und normalisieren
        q_s_star = np.clip(q_s_star, 1e-12, 1.0)
        q_s_star /= q_s_star.sum()
        
        return q_s_star
    else:
        raise RuntimeError(f"Newton-Solver hat keine Lösung gefunden: {solution.message}")
    
def optimize_q_s_iterative(alpha, C, *messages, max_iter=16, tol=1e-4):
    """
    Berechnet q(s) über Fixed-Point Iteration (Standard in Active Inference).
    """
    alpha_sum = np.sum(alpha, axis=0)
    E_A = alpha / alpha_sum
    E_A_log_A = E_A * (sp.digamma(alpha + 1) - sp.digamma(alpha_sum + 1))
    expected_A_log_A_sum = np.sum(E_A_log_A, axis=0)
    
    log_messages = sum(safelog(msg) for msg in messages)
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    num_states = alpha.shape[1]
    q_s = np.ones(num_states) / num_states # Startverteilung
    
    # Fixed-Point Iteration anstelle eines Newton-Solvers
    for _ in range(max_iter):
        q_o = E_A @ q_s
        rho = expected_A_log_A_sum + E_A.T @ (safelog(C) - safelog(q_o))
        q_s_new = softmax(rho + log_messages)
        
        # Abbruchbedingung, falls Konvergenz erreicht ist
        if np.linalg.norm(q_s_new - q_s) < tol:
            q_s = q_s_new
            break
            
        q_s = q_s_new

    # Normalisierung zur absoluten Sicherheit
    q_s = np.clip(q_s, 1e-12, 1.0)
    return q_s / q_s.sum()