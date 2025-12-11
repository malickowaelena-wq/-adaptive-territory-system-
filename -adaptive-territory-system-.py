#!/usr/bin/env python3
"""
АДАПТИВНАЯ СИСТЕМА СТРАТЕГИЧЕСКОГО ТЕРРИТОРИАЛЬНОГО ПЛАНИРОВАНИЯ
Исправленная версия с обработкой ошибок
"""

import sys
import subprocess
import importlib.util
import os
import platform
import traceback

def print_banner():
    """Вывод красивого баннера"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   АДАПТИВНАЯ СИСТЕМА СТРАТЕГИЧЕСКОГО ТЕРРИТОРИАЛЬНОГО ПЛАНИРОВАНИЯ  ║
    ║                                                                   ║
    ║               МАТЕМАТИЧЕСКАЯ МОДЕЛЬ НАРУШЕННЫХ ТЕРРИТОРИЙ         ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print("\033[92m" + banner + "\033[0m")

def check_and_install_packages():
    """
    Проверяет и устанавливает все необходимые пакеты
    """
    print("\n🔍 ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 60)
    
    required_packages = [
        ('numpy', 'numpy>=1.21.0'),
        ('matplotlib', 'matplotlib>=3.5.0'),
        ('seaborn', 'seaborn>=0.11.0'),
        ('scipy', 'scipy>=1.7.0'),
        ('pandas', 'pandas>=1.3.0'),
        ('tqdm', 'tqdm>=4.62.0'),
    ]
    
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ Требуется Python 3.7 или выше")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    pip_command = [sys.executable, "-m", "pip"] if sys.executable else ["pip"]
    
    try:
        subprocess.run([*pip_command, "--version"], check=True, capture_output=True)
        print("✅ Pip доступен")
    except:
        print("❌ Pip не найден")
        return False
    
    installed_packages = []
    missing_packages = []
    
    for package_name, package_spec in required_packages:
        try:
            spec = importlib.util.find_spec(package_name.split('==')[0].split('>=')[0])
            if spec is None:
                raise ImportError
            print(f"✅ {package_name} уже установлен")
            installed_packages.append(package_name)
        except ImportError:
            print(f"❌ {package_name} не найден")
            missing_packages.append(package_spec)
    
    if missing_packages:
        print(f"\n📦 УСТАНОВКА {len(missing_packages)} ОТСУТСТВУЮЩИХ ПАКЕТОВ...")
        
        for i, package_spec in enumerate(missing_packages, 1):
            print(f"\n[{i}/{len(missing_packages)}] Установка: {package_spec}")
            try:
                result = subprocess.run(
                    [*pip_command, "install", package_spec, "--quiet"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"   ✅ Успешно установлен")
                else:
                    base_package = package_spec.split('>=')[0].split('==')[0]
                    print(f"   ⚠️ Проблемы с установкой. Пробуем: {base_package}")
                    
                    result = subprocess.run(
                        [*pip_command, "install", base_package, "--quiet"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        print(f"   ✅ Установлена базовая версия")
                    else:
                        print(f"   ❌ Не удалось установить")
                        return False
                        
            except Exception as e:
                print(f"   ❌ Ошибка: {str(e)}")
                return False
        
        print("\n✅ ВСЕ ПАКЕТЫ УСТАНОВЛЕНЫ!")
    else:
        print("\n✅ ВСЕ НЕОБХОДИМЫЕ ПАКЕТЫ УЖЕ УСТАНОВЛЕНЫ!")
    
    return True

def create_project_structure():
    """Создает структуру проекта"""
    directories = [
        'adaptive_territory_system',
        'adaptive_territory_system/results',
        'adaptive_territory_system/data',
        'adaptive_territory_system/plots',
        'adaptive_territory_system/reports'
    ]
    
    print("\n📁 СОЗДАНИЕ СТРУКТУРЫ ПРОЕКТА")
    print("-" * 60)
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Создана папка: {directory}")
        except Exception as e:
            print(f"⚠️ Не удалось создать {directory}: {e}")
    
    return True

def main_program():
    """
    Основная программа моделирования
    """
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ОСНОВНОЙ ПРОГРАММЫ")
    print("=" * 60)
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        import pandas as pd
        from typing import List, Tuple, Dict
        import json
        from tqdm import tqdm
        
        print("✅ Все библиотеки успешно загружены")
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return
    
    # ============================================================================
    # ИСПРАВЛЕННАЯ ВЕРСИЯ МОДЕЛИ
    # ============================================================================
    
    class SystemParameters:
        """Параметры адаптивной системы"""
        def __init__(self, 
                    S0: float = 100.0,
                    Ti0: float = 50.0,
                    epsilon: float = 0.1,
                    lambda_base: float = 0.08,
                    Delta_t: int = 5,
                    Delta_S: float = 15.0,
                    Delta_Ti: float = 12.0,
                    T: int = 20,
                    collapse_threshold: float = 20.0,
                    binary_threshold: float = 0.7):
            
            self.S0 = S0
            self.Ti0 = Ti0
            self.epsilon = epsilon
            self.lambda_base = lambda_base
            self.Delta_t = Delta_t
            self.Delta_S = Delta_S
            self.Delta_Ti = Delta_Ti
            self.T = T
            self.collapse_threshold = collapse_threshold
            self.binary_threshold = binary_threshold
            self.Ti_threshold = self.binary_threshold * self.Ti0
        
        def get_probabilities(self, error: float = 0.0) -> Dict[str, float]:
            effective_epsilon = min(max(self.epsilon + error * 0.1, 0), 0.9)
            pd = max(0.5 - effective_epsilon / 2, 0.05)
            pn = max(0.5 - effective_epsilon / 2, 0.05)
            pr = min(effective_epsilon, 0.9)
            total = pd + pn + pr
            return {
                'success': pd / total,
                'failure': pn / total,
                'uncertain': pr / total
            }
    
    class AdaptiveTerritorySystem:
        def __init__(self, params: SystemParameters):
            self.params = params
            self.history = []
            self.collapse_risk = 0.0
            
            print(f"\n📊 Параметры системы:")
            print(f"   S₀ = {params.S0:.1f}, Ti₀ = {params.Ti0:.1f}")
            print(f"   ε = {params.epsilon:.2f}, λ = {params.lambda_base:.3f}")
            print(f"   Δt = {params.Delta_t} лет, T = {params.T} лет")
        
        def lambda_effective(self, Ti: float, S: float) -> float:
            lambda_base = self.params.lambda_base
            if Ti < self.params.Ti_threshold:
                lambda_base *= 1.5
            if S < self.params.collapse_threshold * 2:
                collapse_factor = 1 + (self.params.collapse_threshold * 2 - S) / self.params.collapse_threshold
                lambda_base *= collapse_factor
            return lambda_base
        
        def state_degradation(self, S_prev: float, Ti: float, dt: float = 1) -> float:
            lambda_eff = self.lambda_effective(Ti, S_prev)
            return S_prev * np.exp(-lambda_eff * dt)
        
        def human_choice_intervention(self, S: float, accumulated_error: float = 0.0) -> Tuple[float, str, float]:
            probs = self.params.get_probabilities(accumulated_error)
            outcome = np.random.choice(
                ['success', 'failure', 'uncertain'],
                p=[probs['success'], probs['failure'], probs['uncertain']]
            )
            
            delta_error = 0.0
            
            if outcome == 'success':
                new_S = S + self.params.Delta_S
                delta_error = -0.1
            elif outcome == 'failure':
                new_S = S * 0.9
                delta_error = 0.2
            else:
                uncertainty = np.random.uniform(-0.3 * self.params.Delta_S, 0.5 * self.params.Delta_S)
                new_S = max(S + uncertainty, self.params.collapse_threshold * 0.5)
                delta_error = 0.1
                human_factor = np.random.choice(['conservative', 'progressive', 'neutral'])
                self.history.append({
                    'time': len(self.history),
                    'choice': human_factor,
                    'effect': uncertainty
                })
            
            return new_S, outcome, delta_error
        
        def characteristic_development(self, t: int, shocks: List[Tuple[int, float]] = None) -> float:
            Ti = self.params.Ti0 + self.params.Delta_Ti * (t / self.params.T)
            if self.history:
                recent_decisions = [d for d in self.history if d['time'] >= t - 5]
                if recent_decisions:
                    progressive_count = sum(1 for d in recent_decisions if d['choice'] == 'progressive')
                    Ti += progressive_count * 2
            if shocks:
                for shock_time, shock_effect in shocks:
                    if t == shock_time:
                        Ti += shock_effect
            return max(Ti, self.params.Ti0 * 0.3)
        
        def check_collapse_risk(self, S: float, Ti: float) -> float:
            risk = 0.0
            if S < self.params.collapse_threshold:
                risk += (self.params.collapse_threshold - S) / self.params.collapse_threshold
            expected_S = self.params.S0 * (Ti / self.params.Ti0)
            if S < expected_S * 0.5:
                risk += 0.3
            uncertain_count = sum(1 for h in self.history if 'choice' in h)
            risk += min(uncertain_count * 0.05, 0.3)
            self.collapse_risk = min(max(risk, 0), 1)
            return self.collapse_risk
        
        def run_simulation(self, shocks: List[Tuple[int, float]] = None, n_runs: int = 100) -> Dict:
            print(f"\n🔄 Запуск симуляции ({n_runs} прогонов)...")
            
            # Инициализируем результаты
            times = np.arange(0, self.params.T + 1)
            S_trajectories = np.zeros((n_runs, len(times)))
            Ti_trajectories = np.zeros((n_runs, len(times)))
            
            # Для хранения данных разной длины
            outcomes_list = []
            collapse_risks_list = []
            human_choices_list = []
            binary_states_list = []
            
            collapse_count = 0
            
            for run in tqdm(range(n_runs), desc="Прогоны", unit="прог"):
                self.history = []
                self.collapse_risk = 0.0
                
                S_values = np.zeros(len(times))
                Ti_values = np.zeros(len(times))
                outcomes = []
                collapse_risks = []
                human_choices = []
                binary_states = []
                
                accumulated_error = 0.0
                
                S_values[0] = self.params.S0
                Ti_values[0] = self.params.Ti0
                
                for t_idx, t in enumerate(times[1:], 1):
                    # Развитие характеристики Ti
                    Ti_values[t_idx] = self.characteristic_development(t, shocks)
                    
                    # Деградация состояния S
                    S_values[t_idx] = self.state_degradation(S_values[t_idx-1], Ti_values[t_idx])
                    
                    # Проверка на необходимость обновления
                    if t % self.params.Delta_t == 0 and t != 0:
                        new_S, outcome, error_change = self.human_choice_intervention(
                            S_values[t_idx], accumulated_error
                        )
                        S_values[t_idx] = new_S
                        outcomes.append((t, outcome))
                        accumulated_error += error_change
                        accumulated_error = max(min(accumulated_error, 1), -1)
                        
                        if self.history and self.history[-1]['time'] == t:
                            human_choices.append(self.history[-1]['choice'])
                    
                    # Проверка бинарности системы
                    is_binary = len([h for h in self.history if 'choice' in h]) == 0
                    binary_states.append(is_binary)
                    
                    # Оценка риска коллапса
                    risk = self.check_collapse_risk(S_values[t_idx], Ti_values[t_idx])
                    collapse_risks.append(risk)
                    
                    # Проверка на коллапс
                    if risk > 0.8 or S_values[t_idx] < self.params.collapse_threshold * 0.3:
                        S_values[t_idx:] = S_values[t_idx] * 0.5
                        collapse_count += 1
                        break
                
                # Сохраняем траектории
                S_trajectories[run] = S_values
                Ti_trajectories[run] = Ti_values
                
                # Сохраняем данные разной длины в списки
                outcomes_list.append(outcomes)
                collapse_risks_list.append(collapse_risks)
                human_choices_list.append(human_choices)
                binary_states_list.append(binary_states)
            
            # Вычисляем статистики
            S_mean = np.mean(S_trajectories, axis=0)
            S_std = np.std(S_trajectories, axis=0)
            Ti_mean = np.mean(Ti_trajectories, axis=0)
            Ti_std = np.std(Ti_trajectories, axis=0)
            
            collapse_rate = collapse_count / n_runs
            
            print(f"✅ Симуляция завершена. Ставка коллапса: {collapse_rate:.1%}")
            
            return {
                'times': times,
                'S_trajectories': S_trajectories,
                'Ti_trajectories': Ti_trajectories,
                'S_mean': S_mean,
                'S_std': S_std,
                'Ti_mean': Ti_mean,
                'Ti_std': Ti_std,
                'outcomes': outcomes_list,
                'collapse_risks': collapse_risks_list,
                'human_choices': human_choices_list,
                'binary_states': binary_states_list,
                'collapse_rate': collapse_rate
            }
    
    def create_simple_visualization(results: Dict, params: SystemParameters):
        """Создание простых визуализаций"""
        print("\n🎨 СОЗДАНИЕ ГРАФИКОВ...")
        
        times = results['times']
        
        try:
            # 1. Основной график динамики
            plt.figure(figsize=(14, 10))
            
            # График состояния S
            plt.subplot(2, 2, 1)
            plt.plot(times, results['S_mean'], 'b-', linewidth=2, label='Среднее S')
            plt.fill_between(times, 
                            results['S_mean'] - results['S_std'],
                            results['S_mean'] + results['S_std'],
                            alpha=0.3, color='blue')
            plt.axhline(y=params.collapse_threshold, color='r', linestyle='--', label='Порог коллапса')
            plt.xlabel('Время (годы)')
            plt.ylabel('Состояние территории S')
            plt.title('Динамика состояния территории')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # График характеристики Ti
            plt.subplot(2, 2, 2)
            plt.plot(times, results['Ti_mean'], 'g-', linewidth=2, label='Среднее Ti')
            plt.fill_between(times,
                            results['Ti_mean'] - results['Ti_std'],
                            results['Ti_mean'] + results['Ti_std'],
                            alpha=0.3, color='green')
            plt.axhline(y=params.Ti_threshold, color='orange', linestyle=':', label='Бинарный порог')
            plt.xlabel('Время (годы)')
            plt.ylabel('Характеристика Ti')
            plt.title('Развитие характеристик')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # График риска коллапса (средний по всем прогонам)
            plt.subplot(2, 2, 3)
            
            # Вычисляем средний риск коллапса по времени
            max_len = max(len(risks) for risks in results['collapse_risks'])
            collapse_risks_padded = []
            
            for risks in results['collapse_risks']:
                if len(risks) < max_len:
                    padded = list(risks) + [0] * (max_len - len(risks))
                else:
                    padded = risks
                collapse_risks_padded.append(padded)
            
            if collapse_risks_padded:
                collapse_risks_array = np.array(collapse_risks_padded)
                collapse_risk_mean = np.mean(collapse_risks_array, axis=0)
                
                risk_times = np.arange(1, len(collapse_risk_mean) + 1)
                plt.plot(risk_times, collapse_risk_mean, 'r-', linewidth=2)
                plt.fill_between(risk_times, 0, collapse_risk_mean, alpha=0.3, color='red')
                plt.xlabel('Время (годы)')
                plt.ylabel('Риск коллапса')
                plt.title('Динамика риска системного коллапса')
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 1])
            else:
                plt.text(0.5, 0.5, 'Нет данных о риске коллапса', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Динамика риска системного коллапса')
            
            # График распределения конечных состояний
            plt.subplot(2, 2, 4)
            final_S = results['S_trajectories'][:, -1]
            plt.hist(final_S, bins=20, alpha=0.7, color='purple', edgecolor='black')
            plt.axvline(x=np.mean(final_S), color='r', linestyle='-', label=f'Среднее: {np.mean(final_S):.1f}')
            plt.axvline(x=np.median(final_S), color='g', linestyle='--', label=f'Медиана: {np.median(final_S):.1f}')
            plt.xlabel('Конечное состояние S(20)')
            plt.ylabel('Частота')
            plt.title('Распределение конечных состояний')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.suptitle('РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ АДАПТИВНОЙ СИСТЕМЫ', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('adaptive_territory_system/plots/main_results.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Основные графики сохранены")
            
            # 2. Сохраняем данные в CSV
            df = pd.DataFrame({
                'year': times,
                'S_mean': results['S_mean'],
                'S_std': results['S_std'],
                'Ti_mean': results['Ti_mean'],
                'Ti_std': results['Ti_std']
            })
            df.to_csv('adaptive_territory_system/data/simulation_results.csv', index=False)
            print("✅ Данные сохранены в CSV")
            
            # 3. Дополнительный график: сравнение траекторий
            plt.figure(figsize=(12, 8))
            
            # Показываем несколько случайных траекторий
            np.random.seed(42)
            sample_indices = np.random.choice(len(results['S_trajectories']), min(10, len(results['S_trajectories'])), replace=False)
            
            for i, idx in enumerate(sample_indices):
                plt.plot(times, results['S_trajectories'][idx], alpha=0.3, linewidth=1)
            
            plt.plot(times, results['S_mean'], 'k-', linewidth=3, label='Средняя траектория')
            plt.axhline(y=params.collapse_threshold, color='r', linestyle='--', linewidth=2, label='Порог коллапса')
            
            plt.xlabel('Время (годы)')
            plt.ylabel('Состояние территории S')
            plt.title('Примеры отдельных траекторий системы')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig('adaptive_territory_system/plots/trajectories_examples.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print("✅ Дополнительные графики сохранены")
            
            # 4. Создаем текстовый отчет
            report = f"""
ОТЧЕТ ПО РЕЗУЛЬТАТАМ МОДЕЛИРОВАНИЯ
=================================

ПАРАМЕТРЫ СИСТЕМЫ:
- Начальное состояние S₀: {params.S0}
- Начальная характеристика Ti₀: {params.Ti0}
- Влияние социального выбора (ε): {params.epsilon}
- Коэффициент деградации (λ): {params.lambda_base}
- Интервал обновлений (Δt): {params.Delta_t} лет
- Порог коллапса: {params.collapse_threshold}

РЕЗУЛЬТАТЫ:
- Среднее конечное состояние S(20): {results['S_mean'][-1]:.1f} ± {results['S_std'][-1]:.1f}
- Средняя конечная характеристика Ti(20): {results['Ti_mean'][-1]:.1f} ± {results['Ti_std'][-1]:.1f}
- Ставка коллапса: {results['collapse_rate']:.1%}
- Минимальное S(20): {np.min(results['S_trajectories'][:, -1]):.1f}
- Максимальное S(20): {np.max(results['S_trajectories'][:, -1]):.1f}

АНАЛИЗ:
1. Риск коллапса: {'ВЫСОКИЙ (>30%)' if results['collapse_rate'] > 0.3 else 'УМЕРЕННЫЙ' if results['collapse_rate'] > 0.1 else 'НИЗКИЙ'}
2. Устойчивость системы: {'НИЗКАЯ' if results['collapse_rate'] > 0.3 else 'СРЕДНЯЯ' if results['collapse_rate'] > 0.1 else 'ВЫСОКАЯ'}
3. Влияние социального выбора: {'ЗНАЧИТЕЛЬНОЕ' if params.epsilon > 0.2 else 'УМЕРЕННОЕ' if params.epsilon > 0.05 else 'СЛАБОЕ'}

РЕКОМЕНДАЦИИ:
- {'⚠️ СРОЧНО: Увеличить частоту обновлений (уменьшить Δt)' if results['collapse_rate'] > 0.4 else 
   '✅ Рекомендуется: Увеличить частоту обновлений' if results['collapse_rate'] > 0.2 else 
   '✅ Текущая стратегия эффективна'}
- {'⚠️ Снизить влияние социального выбора (уменьшить ε)' if params.epsilon > 0.2 and results['collapse_rate'] > 0.3 else 
   '✅ Оптимальный уровень ε'}
- {'⚠️ Увеличить прирост при обновлении (ΔS)' if results['S_mean'][-1] < params.collapse_threshold * 1.5 else 
   '✅ Прирост ΔS достаточен'}

ДАТА АНАЛИЗА: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
            
            with open('adaptive_territory_system/reports/summary_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            
            print("✅ Отчет сохранен")
            
        except Exception as e:
            print(f"❌ Ошибка при создании визуализации: {e}")
            traceback.print_exc()
    
    # ============================================================================
    # ОСНОВНОЙ БЛОК ЗАПУСКА
    # ============================================================================
    
    try:
        # Создаем параметры системы
        params = SystemParameters(
            S0=100.0,
            Ti0=50.0,
            epsilon=0.1,
            lambda_base=0.08,
            Delta_t=5,
            Delta_S=15.0,
            Delta_Ti=12.0,
            T=20,
            collapse_threshold=20.0,
            binary_threshold=0.7
        )
        
        # Определяем шоки
        shocks = [
            (8, -25),  # Негативный шок в год 8
            (14, +20)  # Позитивный шок в год 14
        ]
        
        # Создаем и запускаем систему
        system = AdaptiveTerritorySystem(params)
        results = system.run_simulation(shocks=shocks, n_runs=100)
        
        # Создаем визуализацию
        create_simple_visualization(results, params)
        
        print("\n" + "=" * 60)
        print("✅ МОДЕЛИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("=" * 60)
        print("\n📁 Результаты сохранены в папках:")
        print("   - adaptive_territory_system/plots/     (графики)")
        print("   - adaptive_territory_system/data/      (данные)")
        print("   - adaptive_territory_system/reports/   (отчеты)")
        print(f"\n📊 Основные результаты:")
        print(f"   • Ставка коллапса: {results['collapse_rate']:.1%}")
        print(f"   • Среднее S(20): {results['S_mean'][-1]:.1f}")
        print(f"   • Среднее Ti(20): {results['Ti_mean'][-1]:.1f}")
        
        # Сохраняем полные результаты
        np.save('adaptive_territory_system/results/S_trajectories.npy', results['S_trajectories'])
        np.save('adaptive_territory_system/results/Ti_trajectories.npy', results['Ti_trajectories'])
        
        print("   • Полные данные сохранены в .npy формате")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА В РАБОТЕ ПРОГРАММЫ: {e}")
        traceback.print_exc()
        return False
    
    return True

# ============================================================================
# ТОЧКА ВХОДА ПРОГРАММЫ
# ============================================================================

if __name__ == "__main__":
    try:
        print_banner()
        
        if not check_and_install_packages():
            print("\n❌ Не удалось установить необходимые зависимости")
            sys.exit(1)
        
        if not create_project_structure():
            print("\n⚠️ Не удалось создать структуру проекта")
        
        success = main_program()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 ПРОГРАММА УСПЕШНО ВЫПОЛНЕНА!")
            print("=" * 60)
            print("\nДля повторного запуска выполните:")
            print("python adaptive_territory_model.py")
        else:
            print("\n❌ Программа завершилась с ошибкой")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        traceback.print_exc()
        sys.exit(1)