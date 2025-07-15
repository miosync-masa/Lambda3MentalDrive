"""
Λ³理論 レジーム粘性テンソル拡張モジュール（修正版）
Regime Viscosity Tensor Extension for Lambda3 Theory

レジーム粘性：状態に「とどまり続ける力」を数理化
- アトラクター盆地（どこに引き寄せられるか）とは独立
- PATH×自己側面の組み合わせで決まる「抜けにくさ」
- 時間依存ではなくトランザクションベースの変化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ========== 基本データ構造 ==========

@dataclass
class ViscosityPattern:
    """高粘性パターンの定義"""
    pattern_id: str
    description: str
    required_paths: Dict[str, float]  # PATH名: 閾値
    required_self_aspects: Dict[str, Tuple[float, float]]  # 側面名: (最小, 最大)
    viscosity_contribution: float  # このパターンが粘性に寄与する量


@dataclass
class InterventionStrategy:
    """介入戦略"""
    transaction_type: str
    description: str
    target_paths: List[str]
    expected_viscosity_reduction: float


# ========== 粘性評価質問システム ==========

class ViscosityAssessment:
    """個体の基準粘性（ν₀）を評価する質問システム"""
    
    def __init__(self):
        self.questions = self._generate_viscosity_questions()
    
    def _generate_viscosity_questions(self) -> List[Dict]:
        """粘性体質判定の質問群"""
        return [
            {
                'id': 'visc_1',
                'text': '嫌なことがあった時、どのくらい引きずりますか？',
                'scale': '1:すぐ忘れる 2:数時間 3:その日中 4:2-3日 5:1週間以上',
                'weight': 1.2,
                'type': 'base_viscosity'
            },
            {
                'id': 'visc_2',
                'text': '良い気分の時、ちょっとしたことで台無しになりやすいですか？',
                'scale': '1:全くない 2:めったに 3:時々 4:よくある 5:非常によくある',
                'weight': 0.8,
                'type': 'mood_fragility'
            },
            {
                'id': 'visc_3',
                'text': '一度落ち込むと、自力で気分を切り替えるのは難しいですか？',
                'scale': '1:簡単に切り替える 2:少し努力すれば 3:かなり努力が必要 4:とても難しい 5:ほぼ不可能',
                'weight': 1.0,
                'type': 'recovery_difficulty'
            },
            {
                'id': 'visc_4',
                'text': '怒りや不満を感じた時、それを手放すまでの時間は？',
                'scale': '1:すぐ手放す 2:10-30分 3:数時間 4:翌日まで 5:何日も続く',
                'weight': 0.9,
                'type': 'anger_persistence'
            },
            {
                'id': 'visc_5',
                'text': '過去の失敗や恥ずかしい記憶が、ふとした時に蘇って気分が沈むことは？',
                'scale': '1:ほとんどない 2:月に1-2回 3:週に1-2回 4:週に数回 5:毎日のように',
                'weight': 0.7,
                'type': 'rumination_tendency'
            }
        ]
    
    def calculate_base_viscosity(self, responses: Dict[str, int]) -> float:
        """質問回答から個体の基準粘性を計算"""
        viscosity_scores = []
        weights = []
        
        for q in self.questions:
            if q['id'] in responses:
                score = (responses[q['id']] - 1) / 4  # 0-1に正規化
                viscosity_scores.append(score)
                weights.append(q['weight'])
        
        if viscosity_scores:
            weighted_avg = sum(s * w for s, w in zip(viscosity_scores, weights)) / sum(weights)
            # 0.2（超低粘性）〜1.0（超高粘性）の範囲にマッピング
            return 0.2 + weighted_avg * 0.8
        else:
            return 0.5  # デフォルト中間値


# ========== レジーム粘性の核心クラス ==========

@dataclass
class RegimeViscosity:
    """レジーム粘性テンソル"""
    
    # 個体基準値（質問から決定）
    base_viscosity: float = 0.5  # ν₀
    
    # PATH×自己側面の相互作用による粘性修正
    path_self_modifiers: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # 高粘性パターンの定義
    viscosity_patterns: List[ViscosityPattern] = field(default_factory=list)
    
    # 現在の環境因子（Lambda3MentalDriverから取得）
    environmental_factors: Dict[str, float] = field(default_factory=lambda: {
        'sleep_factor': 1.0,
        'nutrition_factor': 1.0,
        'recent_event_intensity': 0.0
    })
    
    def __post_init__(self):
        """初期化時にデフォルトパターンを設定"""
        if not self.path_self_modifiers:
            self._initialize_path_self_modifiers()
        if not self.viscosity_patterns:
            self._initialize_viscosity_patterns()
    
    def _initialize_path_self_modifiers(self):
        """PATH×自己側面の相互作用による粘性修正値（より現実的な値に調整）"""
        # 「抜けやすい」組み合わせ（負の値）
        self.path_self_modifiers[('Se', 'core')] = -0.3      # 現実接触×core高
        self.path_self_modifiers[('Te', 'core')] = -0.2      # 実行力×core高
        self.path_self_modifiers[('Fe', 'social')] = -0.2    # 他者接続×social高
        self.path_self_modifiers[('Se', 'social')] = -0.15   # 感覚×社会性
        
        # 「抜けにくい」組み合わせ（正の値）
        self.path_self_modifiers[('Ni', 'shadow')] = 0.4     # 内向直観×shadow
        self.path_self_modifiers[('Si', 'shadow')] = 0.35    # 過去記憶×shadow
        self.path_self_modifiers[('Ti', 'shadow')] = 0.3     # 内的論理×shadow
        self.path_self_modifiers[('Ni', 'ideal')] = 0.25     # 内向直観×理想
        self.path_self_modifiers[('Fi', 'shadow')] = 0.3     # 内的感情×shadow
        
        # Core低下時の各PATHの影響
        self.path_self_modifiers[('Ni', 'core_low')] = 0.3   # Ni過活性×core低
        self.path_self_modifiers[('Ti', 'core_low')] = 0.25  # Ti過活性×core低
        self.path_self_modifiers[('Si', 'core_low')] = 0.2   # Si過活性×core低
        
        # Shadow高×内向PATH（新規追加）
        self.path_self_modifiers[('Ni', 'shadow_high')] = 0.35
        self.path_self_modifiers[('Ti', 'shadow_high')] = 0.3
        self.path_self_modifiers[('Fi', 'shadow_high')] = 0.25
        self.path_self_modifiers[('Si', 'shadow_high')] = 0.3
    
    def _initialize_viscosity_patterns(self):
        """高粘性パターンの定義（より現実的な閾値に調整）"""
        self.viscosity_patterns = [
            ViscosityPattern(
                pattern_id='ni_shadow_rumination',
                description='Ni×Shadow - 内的反芻地獄',
                required_paths={'Ni': 0.4},  # 0.7→0.4に緩和
                required_self_aspects={'shadow': (0.6, 1.0)},  # coreの条件を削除
                viscosity_contribution=0.4
            ),
            ViscosityPattern(
                pattern_id='si_trauma_loop',
                description='Si×Shadow - 過去トラウマループ',
                required_paths={'Si': 0.4},  # 0.6→0.4に緩和
                required_self_aspects={'shadow': (0.65, 1.0)},  # 0.7→0.65に緩和
                viscosity_contribution=0.35
            ),
            ViscosityPattern(
                pattern_id='ti_isolation_fortress',
                description='Ti×低Social - 論理の孤立要塞',
                required_paths={'Ti': 0.45},  # 0.7→0.45に緩和
                required_self_aspects={'social': (0, 0.35)},  # 0.3→0.35に緩和
                viscosity_contribution=0.3
            ),
            ViscosityPattern(
                pattern_id='ideal_reality_gap',
                description='高Ideal×低Core - 理想と現実の断絶',
                required_paths={},  # PATH非依存
                required_self_aspects={'ideal': (0.65, 1.0), 'core': (0, 0.75)},  # 条件緩和
                viscosity_contribution=0.35
            ),
            ViscosityPattern(
                pattern_id='shadow_overwhelm',
                description='Shadow支配 - 自己の影に飲まれる',
                required_paths={},  # PATH非依存
                required_self_aspects={'shadow': (0.75, 1.0)},  # coreとの比較を削除、単純化
                viscosity_contribution=0.45
            ),
            # 新規パターン追加
            ViscosityPattern(
                pattern_id='ni_ti_loop',
                description='Ni-Tiループ - 内的分析地獄',
                required_paths={'Ni': 0.4, 'Ti': 0.4},
                required_self_aspects={},  # 自己側面非依存
                viscosity_contribution=0.35
            ),
            ViscosityPattern(
                pattern_id='low_metacognition_trap',
                description='メタ認知崩壊 - 自己観察不能',
                required_paths={},
                required_self_aspects={'core': (0, 0.5)},  # core低下で発動
                viscosity_contribution=0.3
            )
        ]


# ========== 粘性計算器 ==========

class ViscosityCalculator:
    """レジーム粘性の計算と診断"""
    
    def __init__(self, regime_viscosity: RegimeViscosity):
        self.regime_viscosity = regime_viscosity
    
    def extract_environmental_factors(self, responses: Dict[str, int]) -> Dict[str, float]:
        """Lambda3MentalDriverの既存回答から環境因子を抽出"""
        
        # 睡眠因子（sleep_1: 睡眠時間, sleep_2: 睡眠の質）
        sleep_hours = responses.get('sleep_1', 3)  # デフォルト6-7時間
        sleep_quality = responses.get('sleep_2', 3)  # デフォルト普通
        
        # 睡眠スコア（0.5-1.5: 悪いと粘性増加）
        sleep_score = (sleep_hours + sleep_quality) / 8
        sleep_factor = 1.5 - sleep_score  # 良い睡眠ほど低い値
        
        # 栄養因子（meal_1: 規則性, meal_2: 食事の乱れ）
        meal_regularity = responses.get('meal_1', 3)
        meal_disorder = responses.get('meal_2', 3)
        
        # 栄養スコア（0.5-1.5: 悪いと粘性増加）
        nutrition_score = 2 - (meal_regularity + meal_disorder) / 8
        nutrition_factor = 1.5 - nutrition_score
        
        # 最近のイベント強度（event_1: 喪失体験の大きさ）
        event_intensity = responses.get('event_1', 1)
        recent_event = (event_intensity - 1) / 4  # 0-1に正規化
        
        return {
            'sleep_factor': max(0.5, sleep_factor),
            'nutrition_factor': max(0.5, nutrition_factor),
            'recent_event_intensity': recent_event
        }
    
    def calculate_current_viscosity(self, state) -> Tuple[float, List[str]]:
        """現在の総合粘性と活性パターンを計算（改善版）"""
        
        # stateが粘性を持っていればそれを使う
        if hasattr(state, 'regime_viscosity'):
            regime_viscosity = state.regime_viscosity
        else:
            regime_viscosity = self.regime_viscosity
        
        # 基準値から開始
        total_viscosity = regime_viscosity.base_viscosity
        active_patterns = []
        
        # デバッグ情報を収集
        debug_info = {
            'base': regime_viscosity.base_viscosity,
            'modifiers': [],
            'patterns': []
        }
            
        # 1. PATH×自己側面の組み合わせ効果（改善版）
        for (path, aspect), modifier in self.regime_viscosity.path_self_modifiers.items():
            if aspect == 'core':
                # 通常のcore高状態での効果
                if state.path_states.get(path, 0) > 0.6 and state.Λ_self_aspects['core'] > 0.6:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×{aspect}: +{modifier}")
            elif aspect == 'shadow':
                # Shadow高状態での効果
                if state.path_states.get(path, 0) > 0.4 and state.Λ_self_aspects['shadow'] > 0.6:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×{aspect}: +{modifier}")
            elif aspect == 'shadow_high':
                # Shadow特に高い状態での追加効果
                if state.path_states.get(path, 0) > 0.35 and state.Λ_self_aspects['shadow'] > 0.75:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×shadow_high: +{modifier}")
            elif aspect == 'social':
                # Social高状態での効果
                if state.path_states.get(path, 0) > 0.5 and state.Λ_self_aspects['social'] > 0.5:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×{aspect}: +{modifier}")
            elif aspect == 'ideal':
                # Ideal高状態での効果
                if state.path_states.get(path, 0) > 0.4 and state.Λ_self_aspects['ideal'] > 0.65:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×{aspect}: +{modifier}")
            elif aspect == 'core_low':
                # Core低下時の効果
                if state.path_states.get(path, 0) > 0.4 and state.Λ_self_aspects['core'] < 0.5:
                    total_viscosity += modifier
                    debug_info['modifiers'].append(f"{path}×{aspect}: +{modifier}")
        
        # 2. 特定の高粘性パターンチェック
        for pattern in self.regime_viscosity.viscosity_patterns:
            if self._check_pattern_active(pattern, state):
                total_viscosity += pattern.viscosity_contribution
                active_patterns.append(pattern.pattern_id)
                debug_info['patterns'].append(f"{pattern.pattern_id}: +{pattern.viscosity_contribution}")
        
        # 3. 特殊な状態の追加チェック（新規）
        # Shadow > Coreの場合の追加粘性
        if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
            shadow_dominance = (state.Λ_self_aspects['shadow'] - state.Λ_self_aspects['core']) * 0.5
            total_viscosity += shadow_dominance
            debug_info['modifiers'].append(f"shadow_dominance: +{shadow_dominance:.2f}")
        
        # メタ認知低下による追加粘性
        if hasattr(state, 'metacognition') and state.metacognition < 0.3:
            metacog_penalty = (0.3 - state.metacognition) * 0.8
            total_viscosity += metacog_penalty
            debug_info['modifiers'].append(f"low_metacognition: +{metacog_penalty:.2f}")
        
        # 高ノイズによる追加粘性
        if hasattr(state, 'perception_noise') and state.perception_noise > 0.5:
            noise_penalty = (state.perception_noise - 0.5) * 0.4
            total_viscosity += noise_penalty
            debug_info['modifiers'].append(f"high_noise: +{noise_penalty:.2f}")
        
        # 4. 環境因子の影響
        env_factors = regime_viscosity.environmental_factors
        total_viscosity *= env_factors['sleep_factor']
        total_viscosity *= env_factors['nutrition_factor']
        total_viscosity += env_factors['recent_event_intensity'] * 0.3
        
        # 5. 範囲制限
        total_viscosity = np.clip(total_viscosity, 0.1, 2.0)
        
        # デバッグ出力（必要に応じて）
        # print(f"粘性計算デバッグ: {debug_info}")
        
        return total_viscosity, active_patterns
    
    def _check_pattern_active(self, pattern: ViscosityPattern, state) -> bool:
        """特定のパターンが活性化しているかチェック（改善版）"""
        
        # PATH条件のチェック
        for path, threshold in pattern.required_paths.items():
            if state.path_states.get(path, 0) < threshold:
                return False
        
        # 自己側面条件のチェック
        for aspect, (min_val, max_val) in pattern.required_self_aspects.items():
            aspect_value = state.Λ_self_aspects.get(aspect, 0.5)
            if not (min_val <= aspect_value <= max_val):
                return False
        
        # 特殊パターンの追加チェック
        if pattern.pattern_id == 'low_metacognition_trap':
            # メタ認知も確認
            if hasattr(state, 'metacognition') and state.metacognition > 0.3:
                return False
        
        return True
    
    def diagnose_viscosity_state(self, state, current_viscosity: float, 
                               active_patterns: List[str]) -> Dict:
        """粘性状態の診断"""
        
        # 粘性レベルの判定
        if current_viscosity < 0.3:
            viscosity_level = 'very_low'
            description = '非常に低粘性：気分の切り替えが容易'
        elif current_viscosity < 0.5:
            viscosity_level = 'low'
            description = '低粘性：比較的抜けやすい'
        elif current_viscosity < 0.8:
            viscosity_level = 'moderate'
            description = '中粘性：努力すれば抜けられる'
        elif current_viscosity < 1.2:
            viscosity_level = 'high'
            description = '高粘性：抜けるのが困難'
        else:
            viscosity_level = 'very_high'
            description = '超高粘性：ほぼ抜け出せない状態'
        
        # アクティブパターンの詳細
        pattern_details = []
        for pattern_id in active_patterns:
            pattern = next((p for p in self.regime_viscosity.viscosity_patterns 
                          if p.pattern_id == pattern_id), None)
            if pattern:
                pattern_details.append({
                    'id': pattern.pattern_id,
                    'description': pattern.description,
                    'contribution': pattern.viscosity_contribution
                })
        
        return {
            'viscosity_value': current_viscosity,
            'viscosity_level': viscosity_level,
            'description': description,
            'active_patterns': pattern_details,
            'primary_factors': self._identify_primary_factors(state, active_patterns, current_viscosity)
        }
    
    def _identify_primary_factors(self, state, active_patterns: List[str], 
                                current_viscosity: float) -> List[str]:
        """粘性の主要因を特定（改善版）"""
        factors = []
        
        # 個体要因
        if self.regime_viscosity.base_viscosity > 0.7:
            factors.append('高い基礎粘性体質')
        
        # Shadow支配
        if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
            factors.append('Shadow機能の支配')
        
        # メタ認知崩壊
        if hasattr(state, 'metacognition') and state.metacognition < 0.3:
            factors.append('メタ認知機能の崩壊')
        
        # 高ノイズ
        if hasattr(state, 'perception_noise') and state.perception_noise > 0.5:
            factors.append('知覚ノイズの増大')
        
        # パターン要因
        if 'ni_shadow_rumination' in active_patterns:
            factors.append('内的反芻による固着')
        if 'shadow_overwhelm' in active_patterns:
            factors.append('Shadow機能の圧倒')
        if 'ni_ti_loop' in active_patterns:
            factors.append('Ni-Tiループによる固着')
        
        # 環境要因
        env = self.regime_viscosity.environmental_factors
        if env['sleep_factor'] > 1.2:
            factors.append('睡眠不足による粘性増加')
        if env['nutrition_factor'] > 1.2:
            factors.append('栄養不良による粘性増加')
        if env['recent_event_intensity'] > 0.5:
            factors.append('最近のストレスイベント')
        
        # 身体的枯渇
        if hasattr(state, 'Λ_bod') and state.Λ_bod < 0.4:
            factors.append('身体的エネルギーの枯渇')
        
        return factors


# ========== 介入戦略システム ==========

class ViscosityInterventions:
    """高粘性状態への介入戦略"""
    
    def __init__(self):
        self.intervention_map = self._initialize_intervention_map()
    
    def _initialize_intervention_map(self) -> Dict[str, List[InterventionStrategy]]:
        """パターン別の介入戦略マップ"""
        return {
            'ni_shadow_rumination': [
                InterventionStrategy(
                    transaction_type='se_grounding',
                    description='身体感覚へのグラウンディング（運動、マッサージ、温浴）',
                    target_paths=['Se'],
                    expected_viscosity_reduction=0.3
                ),
                InterventionStrategy(
                    transaction_type='te_small_tasks',
                    description='小さな実行可能タスクの完了で達成感を得る',
                    target_paths=['Te'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='nutrition_protein',
                    description='タンパク質豊富な食事で血糖値を安定化',
                    target_paths=['general'],
                    expected_viscosity_reduction=0.15
                )
            ],
            
            'si_trauma_loop': [
                InterventionStrategy(
                    transaction_type='ne_novelty',
                    description='新しい体験や環境への意図的な露出',
                    target_paths=['Ne'],
                    expected_viscosity_reduction=0.25
                ),
                InterventionStrategy(
                    transaction_type='se_present_moment',
                    description='五感を使った「今ここ」への注意（マインドフルネス）',
                    target_paths=['Se'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='social_new_memory',
                    description='信頼できる他者との新しいポジティブな記憶作り',
                    target_paths=['Fe', 'social'],
                    expected_viscosity_reduction=0.3
                )
            ],
            
            'ti_isolation_fortress': [
                InterventionStrategy(
                    transaction_type='fe_emotional_sharing',
                    description='感情的な体験の共有（サポートグループ、親密な会話）',
                    target_paths=['Fe'],
                    expected_viscosity_reduction=0.3
                ),
                InterventionStrategy(
                    transaction_type='te_externalization',
                    description='思考の外部化（ブログ、プレゼン、教える）',
                    target_paths=['Te'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='body_care',
                    description='身体ケアで思考から離れる（スパ、散歩、ストレッチ）',
                    target_paths=['Se'],
                    expected_viscosity_reduction=0.15
                )
            ],
            
            'ideal_reality_gap': [
                InterventionStrategy(
                    transaction_type='small_wins',
                    description='達成可能な小さな目標での成功体験の積み重ね',
                    target_paths=['Te', 'core'],
                    expected_viscosity_reduction=0.25
                ),
                InterventionStrategy(
                    transaction_type='self_compassion',
                    description='自己への慈しみの練習（セルフコンパッション瞑想）',
                    target_paths=['Fi', 'core'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='sleep_core_restoration',
                    description='十分な睡眠（7-8時間）でcore自己の回復',
                    target_paths=['core'],
                    expected_viscosity_reduction=0.3
                )
            ],
            
            'shadow_overwhelm': [
                InterventionStrategy(
                    transaction_type='shadow_dialogue',
                    description='Shadow側面との意識的な対話（ジャーナリング、夢分析）',
                    target_paths=['shadow'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='creative_expression',
                    description='創造的表現でShadowを昇華（芸術、音楽、執筆）',
                    target_paths=['Fi', 'Ne'],
                    expected_viscosity_reduction=0.25
                ),
                InterventionStrategy(
                    transaction_type='professional_support',
                    description='専門家によるShadowワーク（ユング派分析、深層心理療法）',
                    target_paths=['all'],
                    expected_viscosity_reduction=0.4
                )
            ],
            
            'ni_ti_loop': [
                InterventionStrategy(
                    transaction_type='fe_connection',
                    description='他者との感情的つながりを意図的に作る',
                    target_paths=['Fe'],
                    expected_viscosity_reduction=0.25
                ),
                InterventionStrategy(
                    transaction_type='se_physical_activity',
                    description='激しい身体活動で思考ループを中断',
                    target_paths=['Se'],
                    expected_viscosity_reduction=0.3
                ),
                InterventionStrategy(
                    transaction_type='te_action_plan',
                    description='具体的な行動計画の作成と即実行',
                    target_paths=['Te'],
                    expected_viscosity_reduction=0.2
                )
            ],
            
            'low_metacognition_trap': [
                InterventionStrategy(
                    transaction_type='mindfulness_practice',
                    description='マインドフルネス瞑想で観察者視点を養う',
                    target_paths=['metacognition'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='journaling',
                    description='構造化された日記で自己観察力を回復',
                    target_paths=['metacognition'],
                    expected_viscosity_reduction=0.15
                ),
                InterventionStrategy(
                    transaction_type='feedback_loop',
                    description='信頼できる他者からの定期的フィードバック',
                    target_paths=['Fe', 'metacognition'],
                    expected_viscosity_reduction=0.25
                )
            ],
            
            # 環境要因への介入
            'sleep_deficiency': [
                InterventionStrategy(
                    transaction_type='sleep_hygiene',
                    description='睡眠衛生の改善（定時就寝、スクリーン制限、環境調整）',
                    target_paths=['general'],
                    expected_viscosity_reduction=0.3
                ),
                InterventionStrategy(
                    transaction_type='nap_restoration',
                    description='戦略的な仮眠（20-30分）での一時的回復',
                    target_paths=['general'],
                    expected_viscosity_reduction=0.1
                )
            ],
            
            'nutrition_deficiency': [
                InterventionStrategy(
                    transaction_type='blood_sugar_stability',
                    description='血糖値安定化（低GI食、タンパク質朝食、間食管理）',
                    target_paths=['general'],
                    expected_viscosity_reduction=0.2
                ),
                InterventionStrategy(
                    transaction_type='omega3_supplementation',
                    description='オメガ3脂肪酸の補給（魚、ナッツ、サプリメント）',
                    target_paths=['general'],
                    expected_viscosity_reduction=0.15
                )
            ]
        }
    
    def recommend_interventions(self, diagnosis: Dict, state) -> Dict:
        """診断結果に基づいて介入を推奨"""
        
        recommendations = {
            'immediate_actions': [],
            'supporting_actions': [],
            'maintenance_actions': [],
            'expected_total_reduction': 0.0
        }
        
        # アクティブパターンに対する介入
        for pattern in diagnosis['active_patterns']:
            pattern_id = pattern['id']
            if pattern_id in self.intervention_map:
                strategies = self.intervention_map[pattern_id]
                
                # 最も効果的な介入を即時行動に
                if strategies:
                    primary = max(strategies, key=lambda s: s.expected_viscosity_reduction)
                    recommendations['immediate_actions'].append({
                        'pattern': pattern_id,
                        'intervention': primary,
                        'priority': 'high'
                    })
                    recommendations['expected_total_reduction'] += primary.expected_viscosity_reduction
                    
                    # 残りを支援行動に
                    for strategy in strategies:
                        if strategy != primary:
                            recommendations['supporting_actions'].append({
                                'pattern': pattern_id,
                                'intervention': strategy,
                                'priority': 'medium'
                            })
        
        # 環境要因への対応
        env_factors = state.regime_viscosity.environmental_factors if hasattr(state, 'regime_viscosity') else {}
        
        if env_factors.get('sleep_factor', 1.0) > 1.2:
            sleep_strategies = self.intervention_map.get('sleep_deficiency', [])
            for strategy in sleep_strategies:
                recommendations['supporting_actions'].append({
                    'pattern': 'sleep_deficiency',
                    'intervention': strategy,
                    'priority': 'high'
                })
        
        if env_factors.get('nutrition_factor', 1.0) > 1.2:
            nutrition_strategies = self.intervention_map.get('nutrition_deficiency', [])
            for strategy in nutrition_strategies:
                recommendations['supporting_actions'].append({
                    'pattern': 'nutrition_deficiency',
                    'intervention': strategy,
                    'priority': 'medium'
                })
        
        # 一般的な高粘性への対処（パターンが検出されなかった場合）
        if not recommendations['immediate_actions'] and diagnosis['viscosity_value'] > 0.8:
            # Shadow支配への対処
            if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
                shadow_strategies = self.intervention_map.get('shadow_overwhelm', [])
                if shadow_strategies:
                    recommendations['immediate_actions'].append({
                        'pattern': 'shadow_dominance',
                        'intervention': shadow_strategies[0],
                        'priority': 'high'
                    })
                    recommendations['expected_total_reduction'] += shadow_strategies[0].expected_viscosity_reduction
        
        # 維持行動の推奨
        if diagnosis['viscosity_value'] < 0.5:
            recommendations['maintenance_actions'].extend([
                '現在の低粘性状態を維持するため、定期的な運動を継続',
                '良好な睡眠習慣を保つ',
                '社会的つながりを大切にする'
            ])
        
        # 推奨の要約
        recommendations['summary'] = self._generate_intervention_summary(
            diagnosis, recommendations
        )
        
        return recommendations
    
    def _generate_intervention_summary(self, diagnosis: Dict, 
                                     recommendations: Dict) -> str:
        """介入推奨の要約文を生成"""
        
        viscosity_level = diagnosis['viscosity_level']
        immediate_count = len(recommendations['immediate_actions'])
        expected_reduction = recommendations['expected_total_reduction']
        
        if viscosity_level in ['very_high', 'high']:
            urgency = "緊急に"
            timeframe = "今すぐ"
        elif viscosity_level == 'moderate':
            urgency = "できるだけ早く"
            timeframe = "今週中に"
        else:
            urgency = "予防的に"
            timeframe = "継続的に"
        
        summary = f"現在の粘性レベル（{diagnosis['viscosity_value']:.2f}）は{diagnosis['description']}。"
        
        if immediate_count > 0:
            summary += f"\n{urgency}{immediate_count}つの介入を{timeframe}開始することを推奨。"
            summary += f"これらの介入により、粘性を約{expected_reduction:.1f}ポイント低下させることが期待できます。"
        
        if diagnosis['primary_factors']:
            summary += f"\n主要因（{', '.join(diagnosis['primary_factors'][:2])}）への対処が特に重要です。"
        
        return summary


# ========== 統合インターフェース ==========

class IntegratedViscositySystem:
    """粘性システムの統合インターフェース"""
    
    def __init__(self):
        self.assessment = ViscosityAssessment()
        self.regime_viscosity = RegimeViscosity()
        self.calculator = ViscosityCalculator(self.regime_viscosity)
        self.interventions = ViscosityInterventions()
    
    def initialize_from_responses(self, responses: Dict[str, int]):
        """質問回答から粘性システムを初期化"""
        
        # 基準粘性の設定
        base_viscosity = self.assessment.calculate_base_viscosity(responses)
        self.regime_viscosity.base_viscosity = base_viscosity
        
        # 環境因子の抽出と設定
        env_factors = self.calculator.extract_environmental_factors(responses)
        self.regime_viscosity.environmental_factors.update(env_factors)
    
    def analyze_and_recommend(self, state) -> Dict:
        """状態分析と介入推奨の統合実行"""
        
        # 現在の粘性を計算
        current_viscosity, active_patterns = self.calculator.calculate_current_viscosity(state)
        
        # 診断
        diagnosis = self.calculator.diagnose_viscosity_state(
            state, current_viscosity, active_patterns
        )
        
        # 介入推奨
        recommendations = self.interventions.recommend_interventions(diagnosis, state)
        
        return {
            'diagnosis': diagnosis,
            'recommendations': recommendations,
            'viscosity_metrics': {
                'base': self.regime_viscosity.base_viscosity,
                'current': current_viscosity,
                'reduction_potential': recommendations['expected_total_reduction']
            }
        }
    
    def get_viscosity_questions(self) -> List[Dict]:
        """粘性評価質問を取得"""
        return self.assessment.questions


# ========== デモンストレーション関数 ==========

def demonstrate_viscosity_system():
    """粘性システムのデモンストレーション"""
    
    print("=== Λ³理論 レジーム粘性解析デモ ===\n")
    
    # サンプル回答（高粘性体質の例）
    sample_responses = {
        # 粘性体質質問
        'visc_1': 4,  # 2-3日引きずる
        'visc_2': 4,  # よく気分が台無しになる
        'visc_3': 4,  # 切り替えがとても難しい
        'visc_4': 4,  # 翌日まで怒りが続く
        'visc_5': 5,  # 毎日のように過去が蘇る
        
        # 環境要因（Lambda3MentalDriverから）
        'sleep_1': 4,  # 5-6時間睡眠
        'sleep_2': 4,  # 睡眠の質が悪い
        'meal_1': 3,   # 時々不規則
        'meal_2': 3,   # 時々食事を抜く
        'event_1': 3   # 中程度のストレスイベント
    }
    
    # 仮想的な状態（Shadow支配パターン）
    class MockState:
        def __init__(self):
            self.path_states = {
                'Ni': 0.42,  # 中程度のNi
                'Te': 0.46,
                'Fi': 0.36,
                'Se': 0.27,  # 低Se
                'Ne': 0.46,
                'Ti': 0.46,
                'Fe': 0.51,
                'Si': 0.46
            }
            self.Λ_self_aspects = {
                'core': 0.72,
                'ideal': 0.70,
                'social': 0.26,  # 低social
                'shadow': 0.79   # 高shadow
            }
            self.metacognition = 0.19  # 低メタ認知
            self.perception_noise = 0.57  # 高ノイズ
            self.Λ_bod = 0.29  # 低身体エネルギー
    
    # システム初期化
    system = IntegratedViscositySystem()
    system.initialize_from_responses(sample_responses)
    
    # 状態作成
    state = MockState()
    state.regime_viscosity = system.regime_viscosity
    
    # 分析実行
    result = system.analyze_and_recommend(state)
    
    # 結果表示
    print(f"【基準粘性（ν₀）】")
    print(f"  個体値: {result['viscosity_metrics']['base']:.2f}")
    print(f"  （質問回答から「引きずりやすい体質」と判定）\n")
    
    print(f"【現在の粘性状態】")
    diagnosis = result['diagnosis']
    print(f"  総合粘性: {diagnosis['viscosity_value']:.2f}")
    print(f"  レベル: {diagnosis['description']}")
    
    if diagnosis['active_patterns']:
        print(f"\n  アクティブな高粘性パターン:")
        for pattern in diagnosis['active_patterns']:
            print(f"    - {pattern['description']} (+{pattern['contribution']:.2f})")
    
    if diagnosis['primary_factors']:
        print(f"\n  主要因:")
        for factor in diagnosis['primary_factors']:
            print(f"    - {factor}")
    
    print(f"\n【推奨される介入】")
    recommendations = result['recommendations']
    
    if recommendations['immediate_actions']:
        print(f"\n  《即時行動》")
        for action in recommendations['immediate_actions']:
            intervention = action['intervention']
            print(f"    ◆ {intervention.description}")
            print(f"      期待される粘性低下: -{intervention.expected_viscosity_reduction:.2f}")
    
    if recommendations['supporting_actions']:
        print(f"\n  《支援行動》")
        for action in recommendations['supporting_actions'][:3]:  # 最初の3つ
            intervention = action['intervention']
            print(f"    ・ {intervention.description}")
    
    print(f"\n【介入効果の見込み】")
    print(f"  現在の粘性: {diagnosis['viscosity_value']:.2f}")
    print(f"  介入後の予測: {diagnosis['viscosity_value'] - recommendations['expected_total_reduction']:.2f}")
    print(f"  （{recommendations['expected_total_reduction']:.2f}ポイントの低下）")
    
    print(f"\n【要約】")
    print(f"  {recommendations['summary']}")
    
    return system, result


# 実行例
if __name__ == "__main__":
    system, result = demonstrate_viscosity_system()
