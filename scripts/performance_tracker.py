# scripts/performance_tracker.py
import json
import os
from datetime import datetime, timedelta
import pandas as pd

class PerformanceTracker:
    def __init__(self):
        self.results_file = 'logs/trading_results.json'
        self.performance_file = 'logs/performance_summary.json'
        
    def log_trade_result(self, pair, entry_signal, entry_price, exit_price, exit_reason, position_size):
        """Registrar resultado de un trade"""
        try:
            # Calcular PnL
            if entry_signal == 'BUY':
                pnl_pips = (exit_price - entry_price) * 10000
            else:  # SELL
                pnl_pips = (entry_price - exit_price) * 10000
            
            # Ajustar para pares con JPY
            if 'JPY' in pair:
                pnl_pips = pnl_pips / 100
            
            # Calcular PnL en USD (aproximado)
            pnl_usd = pnl_pips * position_size * 10  # Estimación
            
            trade_result = {
                'timestamp': datetime.now().isoformat(),
                'pair': pair,
                'signal': entry_signal,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'exit_reason': exit_reason,
                'position_size': position_size,
                'pnl_pips': round(pnl_pips, 1),
                'pnl_usd': round(pnl_usd, 2),
                'win': pnl_pips > 0
            }
            
            # Cargar resultados existentes
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            
            results.append(trade_result)
            
            # Guardar
            os.makedirs('logs', exist_ok=True)
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Trade result logged: {pair} {pnl_pips:.1f} pips")
            
            # Actualizar resumen de performance
            self.update_performance_summary()
            
        except Exception as e:
            print(f"Error logging trade result: {e}")
    
    def update_performance_summary(self):
        """Actualizar resumen de performance"""
        try:
            if not os.path.exists(self.results_file):
                return
            
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            
            if not results:
                return
            
            # Calcular métricas
            total_trades = len(results)
            winning_trades = len([r for r in results if r['win']])
            losing_trades = total_trades - winning_trades
            
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            total_pnl = sum(r['pnl_usd'] for r in results)
            avg_win = sum(r['pnl_usd'] for r in results if r['win']) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum(r['pnl_usd'] for r in results if not r['win']) / losing_trades if losing_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(r['pnl_usd'] for r in results if r['win'])
            gross_loss = abs(sum(r['pnl_usd'] for r in results if not r['win']))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Máximo drawdown
            running_pnl = 0
            peak = 0
            max_drawdown = 0
            
            for result in results:
                running_pnl += result['pnl_usd']
                if running_pnl > peak:
                    peak = running_pnl
                drawdown = peak - running_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Último mes
            one_month_ago = datetime.now() - timedelta(days=30)
            recent_results = [
                r for r in results 
                if datetime.fromisoformat(r['timestamp']) > one_month_ago
            ]
            
            recent_pnl = sum(r['pnl_usd'] for r in recent_results)
            recent_trades = len(recent_results)
            recent_win_rate = (len([r for r in recent_results if r['win']]) / recent_trades * 100) if recent_trades > 0 else 0
            
            performance_summary = {
                'updated': datetime.now().isoformat(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate_percent': round(win_rate, 2),
                'total_pnl_usd': round(total_pnl, 2),
                'average_win_usd': round(avg_win, 2),
                'average_loss_usd': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown_usd': round(max_drawdown, 2),
                'last_30_days': {
                    'trades': recent_trades,
                    'pnl_usd': round(recent_pnl, 2),
                    'win_rate_percent': round(recent_win_rate, 2)
                },
                'pair_performance': self.calculate_pair_performance(results)
            }
            
            with open(self.performance_file, 'w') as f:
                json.dump(performance_summary, f, indent=2)
            
            print(f"Performance updated: {total_trades} trades, {win_rate:.1f}% win rate, ${total_pnl:.2f} PnL")
            
        except Exception as e:
            print(f"Error updating performance summary: {e}")
    
    def calculate_pair_performance(self, results):
        """Calcular performance por par"""
        try:
            pair_stats = {}
            
            for result in results:
                pair = result['pair']
                
                if pair not in pair_stats:
                    pair_stats[pair] = {
                        'trades': 0,
                        'wins': 0,
                        'pnl': 0
                    }
                
                pair_stats[pair]['trades'] += 1
                if result['win']:
                    pair_stats[pair]['wins'] += 1
                pair_stats[pair]['pnl'] += result['pnl_usd']
            
            # Calcular win rates
            for pair, stats in pair_stats.items():
                stats['win_rate'] = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                stats['pnl'] = round(stats['pnl'], 2)
                stats['win_rate'] = round(stats['win_rate'], 2)
            
            return pair_stats
            
        except Exception as e:
            print(f"Error calculating pair performance: {e}")
            return {}
    
    def generate_daily_report(self):
        """Generar reporte diario"""
        try:
            if not os.path.exists(self.performance_file):
                return
            
            with open(self.performance_file, 'r') as f:
                performance = json.load(f)
            
            # Crear reporte
            report = f"""
Trading Bot Daily Report - {datetime.now().strftime('%Y-%m-%d')}

OVERALL PERFORMANCE:
- Total Trades: {performance['total_trades']}
- Win Rate: {performance['win_rate_percent']}%
- Total PnL: ${performance['total_pnl_usd']}
- Profit Factor: {performance['profit_factor']}
- Max Drawdown: ${performance['max_drawdown_usd']}

LAST 30 DAYS:
- Trades: {performance['last_30_days']['trades']}
- PnL: ${performance['last_30_days']['pnl_usd']}
- Win Rate: {performance['last_30_days']['win_rate_percent']}%

TOP PERFORMING PAIRS:
"""
            
            # Agregar performance por pares
            sorted_pairs = sorted(
                performance['pair_performance'].items(),
                key=lambda x: x[1]['pnl'],
                reverse=True
            )
            
            for pair, stats in sorted_pairs[:5]:
                report += f"- {pair}: ${stats['pnl']} ({stats['win_rate']}% win rate)\n"
            
            # Guardar reporte
            report_file = f"logs/daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"Daily report generated: {report_file}")
            
        except Exception as e:
            print(f"Error generating daily report: {e}")

if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.update_performance_summary()
    tracker.generate_daily_report()
