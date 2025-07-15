import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# ユング8機能ベースPATH
BASE_PATHS = ['Ne', 'Ni', 'Te', 'Ti', 'Fe', 'Fi', 'Se', 'Si']

@dataclass
class MBTIProfile:
    """MBTIプロファイル"""
    type_code: str  # e.g., "INTJ"
    dominant: str   # e.g., "Ni"
    auxiliary: str  # e.g., "Te"
    tertiary: str   # e.g., "Fi"
    inferior: str   # e.g., "Se"

@dataclass
class EmotionalState:
    """感情状態"""
    joy: float = 0.5
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    disgust: float = 0.0

    def dominant_emotion(self):
        emotions = {
            'joy': self.joy, 'sadness': self.sadness,
            'anger': self.anger, 'fear': self.fear,
            'surprise': self.surprise, 'disgust': self.disgust
        }
        return max(emotions.items(), key=lambda x: x[1])

    def to_vector(self):
        return np.array([self.joy, self.sadness, self.anger,
                        self.fear, self.surprise, self.disgust])

@dataclass
class Desire:
    """欲求/目標"""
    achievement: float = 0.5  # 達成欲
    affiliation: float = 0.5  # 親和欲
    power: float = 0.5       # 支配欲
    security: float = 0.5    # 安定欲
    autonomy: float = 0.5    # 自律欲

    def to_vector(self):
        return np.array([self.achievement, self.affiliation,
                        self.power, self.security, self.autonomy])

# 基本Lambda3State
class Lambda3State:
    """基本Λ³状態"""
    def __init__(self):
        # 基本テンソル
        self.Λ_self = 0.8
        self.σs = 0.5
        self.ρT = 0.6
        self.ΛF = np.array([0.7, 0.3, 0.5])
        self.Λ_bod = 0.7

        # 拡張要素
        self.emotional_state = EmotionalState()
        self.current_desire = Desire()

        # 学習メモリ（PATH×結果の履歴）
        self.path_outcomes = {path: {'positive': 0, 'negative': 0}
                             for path in BASE_PATHS}
        self.path_memory = {path: 0.5 for path in BASE_PATHS}

    def calculate_path_affinity(self, path: str) -> float:
        """学習履歴に基づくPATH親和性"""
        outcomes = self.path_outcomes[path]
        total = outcomes['positive'] + outcomes['negative']
        if total == 0:
            return 0.5
        return outcomes['positive'] / total

    def update_learning(self, path: str, outcome: float):
        """強化学習的な更新"""
        if outcome > 0:
            self.path_outcomes[path]['positive'] += outcome
        else:
            self.path_outcomes[path]['negative'] += abs(outcome)

        # メモリ更新
        self.path_memory[path] = self.calculate_path_affinity(path)

# 発達段階の定義
class DevelopmentalStage(Enum):
    EXPLORATION = "exploration"  # 探索期
    FORMATION = "formation"      # 形成期
    STABILIZATION = "stabilization"  # 安定期
    INTEGRATION = "integration"  # 統合期
    CRISIS = "crisis"           # 危機期

# PATH嗜好の状態
class PathPreference(Enum):
    AVOIDED = "avoided"         # 回避
    RELUCTANT = "reluctant"     # 消極的
    NEUTRAL = "neutral"         # 中立
    PREFERRED = "preferred"     # 好む
    DEPENDENT = "dependent"     # 依存

@dataclass
class SocialFeedback:
    """社会的フィードバック"""
    source_id: str              # フィードバック元
    valence: float             # -1.0(否定的) ~ 1.0(肯定的)
    intensity: float           # 0.0 ~ 1.0(強度)
    authenticity: float        # 0.0 ~ 1.0(真正性)
    category: str              # 'approval', 'criticism', 'support', etc.

@dataclass
class Event:
    """不確実な出来事"""
    event_type: str            # 'loss', 'opportunity', 'conflict', etc.
    impact: float              # -1.0 ~ 1.0
    uncertainty: float         # 0.0 ~ 1.0(不確実性)
    duration: int              # 影響が続く期間

class AdvancedLambda3State(Lambda3State):
    """拡張版Λ³状態（長期発達対応）"""
    def __init__(self):
        super().__init__()

        # 自己定義の多面性
        self.Λ_self_aspects = {
            'core': 0.8,           # 核となる自己
            'ideal': 0.7,          # 理想自己
            'social': 0.6,         # 社会的自己
            'shadow': 0.3          # 影の自己
        }

        # 長期的なPATH嗜好
        self.path_preferences = {
            path: PathPreference.NEUTRAL for path in BASE_PATHS
        }
        self.path_trauma_count = {path: 0 for path in BASE_PATHS}

        # 発達段階
        self.developmental_stage = DevelopmentalStage.FORMATION
        self.stage_duration = 0

        # 社会的フィードバック履歴
        self.social_feedback_history = []

        # 認識ノイズ
        self.perception_noise = 0.1

        # メタ認知能力
        self.metacognition = 0.5

    def update_self_definition(self, experience_type: str, outcome: float):
        """自己定義の多面的更新"""
        # 核となる自己への影響
        if abs(outcome) > 0.7:  # 強い経験
            self.Λ_self_aspects['core'] += outcome * 0.02

        # 理想自己と現実自己のギャップ
        gap = self.Λ_self_aspects['ideal'] - self.Λ_self_aspects['core']
        if gap > 0.5:  # 大きなギャップ
            self.perception_noise += 0.01  # 認知的不協和

        # 影の自己の成長
        if outcome < -0.5:
            self.Λ_self_aspects['shadow'] = min(0.8,
                self.Λ_self_aspects['shadow'] + 0.05)

    def process_social_feedback(self, feedback: SocialFeedback):
        """社会的フィードバックの処理"""
        self.social_feedback_history.append(feedback)

        # 社会的自己の更新
        impact = feedback.valence * feedback.intensity * feedback.authenticity
        self.Λ_self_aspects['social'] += impact * 0.03

        # メタ認知への影響
        if feedback.authenticity > 0.7:
            self.metacognition = min(1.0, self.metacognition + 0.02)

    def check_developmental_transition(self):
        """発達段階の移行チェック（改良版）"""
        self.stage_duration += 1

        # 現在の状態に基づく移行条件
        if self.developmental_stage == DevelopmentalStage.EXPLORATION:
            if self.stage_duration > 20 and self.metacognition > 0.6:
                self.developmental_stage = DevelopmentalStage.FORMATION
                self.stage_duration = 0
                print(f"  → 発達段階移行: FORMATION（形成期）へ")

        elif self.developmental_stage == DevelopmentalStage.FORMATION:
            stable_paths = sum(1 for p in self.path_preferences.values()
                             if p in [PathPreference.PREFERRED, PathPreference.NEUTRAL])
            if stable_paths > 5 and self.stage_duration > 20 and self.metacognition > 0.55:
                self.developmental_stage = DevelopmentalStage.STABILIZATION
                self.stage_duration = 0
                print(f"  → 発達段階移行: STABILIZATION（安定期）へ")

        elif self.developmental_stage == DevelopmentalStage.STABILIZATION:
            # 過剰適応チェック
            preferred_all = sum(1 for p in self.path_preferences.values()
                              if p == PathPreference.PREFERRED)
            if preferred_all >= 7:  # ほぼ全てを好む = 過剰適応
                self.developmental_stage = DevelopmentalStage.CRISIS
                self.stage_duration = 0
                print(f"  → 発達段階移行: CRISIS（危機期）へ - 過剰適応による")

        elif self.developmental_stage == DevelopmentalStage.CRISIS:
            # 回復条件
            if self.stage_duration > 30 and self.metacognition > 0.7:
                # Shadow統合チェック
                if self.Λ_self_aspects['core'] > 0:  # ゼロ除算を防ぐ
                    shadow_integration = self.Λ_self_aspects['shadow'] / self.Λ_self_aspects['core']
                    if shadow_integration < 1.2:  # Shadowが適度に統合された
                        self.developmental_stage = DevelopmentalStage.INTEGRATION
                        self.stage_duration = 0
                        print(f"  → 発達段階移行: INTEGRATION（統合期）へ")

        # トラウマによる危機への移行（どの段階からでも可能）
        trauma_total = sum(self.path_trauma_count.values())
        if trauma_total > 10 and self.developmental_stage not in [DevelopmentalStage.CRISIS, DevelopmentalStage.INTEGRATION]:
            self.developmental_stage = DevelopmentalStage.CRISIS
            self.stage_duration = 0
            print(f"  → 発達段階移行: CRISIS（危機期）へ - トラウマ蓄積による")

@dataclass
class MBTIEmotionalPattern:
    """MBTIタイプ特有の感情・欲求パターン"""
    type_code: str
    emotional_distortions: Dict[str, float]
    desire_distortions: Dict[str, float]

class MBTIEmotionalDynamics:
    """MBTIタイプごとの感情・欲求の歪みパターン"""

    def __init__(self):
        self.patterns = {
            'INTJ': MBTIEmotionalPattern(
                type_code='INTJ',
                emotional_distortions={
                    # INTJは感情を抑圧しがち
                    'joy': -0.3,      # 喜びの表現を抑制
                    'sadness': -0.2,  # 悲しみを隠す
                    'anger': 0.4,     # 内に溜め込んだ怒り
                    'fear': 0.3,      # 失敗への恐れ（隠している）
                    'surprise': -0.2, # 予測を重視するため驚きを嫌う
                    'disgust': 0.2    # 非効率への嫌悪
                },
                desire_distortions={
                    'achievement': 0.4,   # 過度の達成欲求
                    'affiliation': -0.4,  # 親和欲求の抑圧
                    'power': 0.3,         # コントロール欲求
                    'security': -0.2,     # 表面的には安定を軽視
                    'autonomy': 0.5       # 強い自律欲求
                }
            ),
            'INFJ': MBTIEmotionalPattern(
                type_code='INFJ',
                emotional_distortions={
                    'joy': -0.2,      # 控えめな喜び
                    'sadness': 0.3,   # 他者の苦しみへの共感
                    'anger': -0.4,    # 怒りの抑圧
                    'fear': 0.2,      # 対立への恐れ
                    'surprise': -0.1,
                    'disgust': -0.2   # 否定的感情の抑制
                },
                desire_distortions={
                    'achievement': 0.2,
                    'affiliation': 0.4,   # 強い親和欲求
                    'power': -0.4,        # 権力欲求の否定
                    'security': 0.3,      # 調和への欲求
                    'autonomy': -0.2      # 自己犠牲傾向
                }
            ),
            # 他のタイプも追加可能
        }

    def apply_mbti_distortions(self, state: AdvancedLambda3State,
                              mbti_type: str,
                              stress_level: float = 0.5):
        """MBTIタイプ特有の歪みを状態に適用"""

        if mbti_type not in self.patterns:
            return

        pattern = self.patterns[mbti_type]

        # ストレスレベルに応じて歪みを増幅
        amplification = 1.0 + (stress_level - 0.5) * 2.0  # 0.5で標準、1.0で3倍

        # 感情状態の歪み適用
        for emotion, distortion in pattern.emotional_distortions.items():
            current_value = getattr(state.emotional_state, emotion)
            # 歪みを適用（0-1の範囲に収める）
            new_value = np.clip(
                current_value + distortion * amplification,
                0.0, 1.0
            )
            setattr(state.emotional_state, emotion, new_value)

        # 欲求の歪み適用
        for desire, distortion in pattern.desire_distortions.items():
            current_value = getattr(state.current_desire, desire)
            new_value = np.clip(
                current_value + distortion * amplification,
                0.0, 1.0
            )
            setattr(state.current_desire, desire, new_value)

    def detect_pathological_patterns(self, state: AdvancedLambda3State,
                                   mbti_type: str) -> Dict:
        """病理的パターンの検出"""

        pathological_indicators = []

        # INTJ特有のパターン
        if mbti_type == 'INTJ':
            # 感情の過度な抑圧
            emotion_sum = (state.emotional_state.joy +
                          state.emotional_state.sadness +
                          state.emotional_state.anger)
            if emotion_sum < 0.3:
                pathological_indicators.append({
                    'pattern': 'emotional_suppression',
                    'severity': 1.0 - emotion_sum,
                    'description': '感情の過度な抑圧'
                })

            # 達成欲求と親和欲求の極端な不均衡
            achievement_affiliation_ratio = (
                state.current_desire.achievement /
                (state.current_desire.affiliation + 0.1)  # ゼロ除算防止
            )
            if achievement_affiliation_ratio > 3.0:
                pathological_indicators.append({
                    'pattern': 'achievement_isolation',
                    'severity': min(1.0, achievement_affiliation_ratio / 5.0),
                    'description': '達成欲求による孤立'
                })

            # 自律欲求の過剰
            if state.current_desire.autonomy > 0.8:
                pathological_indicators.append({
                    'pattern': 'hyper_independence',
                    'severity': state.current_desire.autonomy,
                    'description': '過度の独立性・他者拒絶'
                })

        return {
            'indicators': pathological_indicators,
            'total_severity': sum(ind['severity'] for ind in pathological_indicators),
            'primary_pattern': max(pathological_indicators,
                                 key=lambda x: x['severity'])['pattern']
                                 if pathological_indicators else None
        }

class MBTIDepressionAssessment:
    """MBTI特異的なうつ・暴走状態の評価システム"""

    def __init__(self):
        # MBTI機能スタック定義
        self.mbti_stacks = {
            'INTJ': MBTIProfile('INTJ', 'Ni', 'Te', 'Fi', 'Se'),
            'INFJ': MBTIProfile('INFJ', 'Ni', 'Fe', 'Ti', 'Se'),
            'ENTJ': MBTIProfile('ENTJ', 'Te', 'Ni', 'Se', 'Fi'),
            'ENFJ': MBTIProfile('ENFJ', 'Fe', 'Ni', 'Se', 'Ti'),
        }

    def generate_intj_assessment(self) -> Dict:
        """INTJ特有の暴走状態・シャドウ評価質問"""

        questions = {
            # Ni（主機能）の過剰使用
            'ni_overuse': [
                {
                    'id': 'ni_1',
                    'text': '最近、将来の悪い結果ばかり想像して、現実的な行動が取れなくなっていますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'Ni',
                    'pattern': 'overuse'
                },
                {
                    'id': 'ni_2',
                    'text': '頭の中で完璧な計画やビジョンを作りすぎて、実行に移せないことが増えていますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Ni',
                    'pattern': 'overuse'
                },
                {
                    'id': 'ni_3',
                    'text': '他人には理解できない複雑な内的世界に閉じこもることが多いですか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.6,
                    'path': 'Ni',
                    'pattern': 'isolation'
                }
            ],

            # Te（補助機能）の機能不全
            'te_dysfunction': [
                {
                    'id': 'te_1',
                    'text': '以前は得意だった効率的な問題解決が、最近できなくなっていますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Te',
                    'pattern': 'suppression'
                },
                {
                    'id': 'te_2',
                    'text': '論理的に考えようとすると、かえって混乱してしまうことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.6,
                    'path': 'Te',
                    'pattern': 'dysfunction'
                }
            ],

            # Fi（第三機能）の不健全な発露
            'fi_eruption': [
                {
                    'id': 'fi_1',
                    'text': '普段は表に出さない個人的な感情が、制御できずに爆発することがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'Fi',
                    'pattern': 'eruption'
                },
                {
                    'id': 'fi_2',
                    'text': '自分の価値観に過度にこだわり、他人を批判的に見てしまうことが増えましたか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.6,
                    'path': 'Fi',
                    'pattern': 'rigidity'
                }
            ],

            # Se（劣等機能）の暴走
            'se_inferior_grip': [
                {
                    'id': 'se_1',
                    'text': '普段と違って、過度に感覚的刺激（食べ物、買い物、娯楽）に耽ることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 1.0,  # 劣等機能は重要
                    'path': 'Se',
                    'pattern': 'inferior_grip'
                },
                {
                    'id': 'se_2',
                    'text': '衝動的で無計画な行動を取ってしまい、後で後悔することが増えましたか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.9,
                    'path': 'Se',
                    'pattern': 'inferior_grip'
                },
                {
                    'id': 'se_3',
                    'text': '身体的な不調（頭痛、胃痛など）に過度に注目してしまうことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Se',
                    'pattern': 'somatic'
                }
            ],

            # 全体的なバランス崩壊
            'general_imbalance': [
                {
                    'id': 'gen_1',
                    'text': '自分が自分でないような感覚に襲われることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'general',
                    'pattern': 'dissociation'
                },
                {
                    'id': 'gen_2',
                    'text': '以前の自分なら簡単にできたことが、できなくなっていますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'general',
                    'pattern': 'regression'
                }
            ],

            # 生活習慣・身体状態
            'lifestyle_physical': [
                {
                    'id': 'sleep_1',
                    'text': '平均的な睡眠時間はどのくらいですか？',
                    'scale': '1:8時間以上 2:7-8時間 3:6-7時間 4:5-6時間 5:5時間未満',
                    'weight': 1.0,
                    'path': 'physical',
                    'pattern': 'sleep_deprivation'
                },
                {
                    'id': 'sleep_2',
                    'text': '睡眠の質はどうですか？（途中で目が覚める、寝付きが悪いなど）',
                    'scale': '1:とても良い 2:良い 3:普通 4:悪い 5:とても悪い',
                    'weight': 0.8,
                    'path': 'physical',
                    'pattern': 'sleep_quality'
                },
                {
                    'id': 'meal_1',
                    'text': '規則正しい食事を取れていますか？',
                    'scale': '1:常に規則的 2:ほぼ規則的 3:時々不規則 4:よく不規則 5:ほとんど不規則',
                    'weight': 0.7,
                    'path': 'physical',
                    'pattern': 'meal_irregularity'
                },
                {
                    'id': 'meal_2',
                    'text': '食事を抜いたり、過食したりすることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'physical',
                    'pattern': 'eating_disorder'
                }
            ],

            # シャドウ機能の評価（INTJのシャドウ: Ne-Ti-Fe-Si）
            'shadow_ne': [
                {
                    'id': 'sh_ne_1',
                    'text': '最近、散漫で収拾のつかないアイデアに振り回されることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'Ne',
                    'pattern': 'shadow_eruption'
                },
                {
                    'id': 'sh_ne_2',
                    'text': '普段と違って、あらゆる可能性を同時に追いかけて混乱することがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Ne',
                    'pattern': 'shadow_chaos'
                }
            ],

            'shadow_ti': [
                {
                    'id': 'sh_ti_1',
                    'text': '自分の論理に過度にこだわり、他者の視点を完全に無視してしまうことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Ti',
                    'pattern': 'shadow_rigidity'
                },
                {
                    'id': 'sh_ti_2',
                    'text': '細かい矛盾にとらわれて、全体像を見失うことが増えましたか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.6,
                    'path': 'Ti',
                    'pattern': 'shadow_obsession'
                }
            ],

            'shadow_fe': [
                {
                    'id': 'sh_fe_1',
                    'text': '他人の感情に過度に振り回され、自分を見失うことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'path': 'Fe',
                    'pattern': 'shadow_absorption'
                },
                {
                    'id': 'sh_fe_2',
                    'text': '表面的な調和を保とうとして、本音を言えなくなっていますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.6,
                    'path': 'Fe',
                    'pattern': 'shadow_facade'
                }
            ],

            'shadow_si': [
                {
                    'id': 'sh_si_1',
                    'text': '過去の失敗や嫌な記憶に囚われて前に進めないことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.9,
                    'path': 'Si',
                    'pattern': 'shadow_rumination'
                },
                {
                    'id': 'sh_si_2',
                    'text': '身体の小さな不調に過度に注目し、不安になることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'path': 'Si',
                    'pattern': 'shadow_hypochondria'
                }
            ]
        }

        return questions

    def calculate_dysfunction_scores(self, responses: Dict[str, int]) -> Dict:
        """回答から機能不全スコアを計算"""

        questions = self.generate_intj_assessment()

        # PATHごとのスコア集計
        path_scores = {
            'Ni': {'overuse': 0, 'total_weight': 0},
            'Te': {'dysfunction': 0, 'total_weight': 0},
            'Fi': {'eruption': 0, 'total_weight': 0},
            'Se': {'inferior_grip': 0, 'total_weight': 0},
            'Ne': {'shadow': 0, 'total_weight': 0},  # シャドウ機能
            'Ti': {'shadow': 0, 'total_weight': 0},  # シャドウ機能
            'Fe': {'shadow': 0, 'total_weight': 0},  # シャドウ機能
            'Si': {'shadow': 0, 'total_weight': 0},  # シャドウ機能
            'general': {'imbalance': 0, 'total_weight': 0},
            'physical': {'dysfunction': 0, 'total_weight': 0}  # 身体的要因
        }

        # 各質問カテゴリのスコア計算
        for category, question_list in questions.items():
            for q in question_list:
                if q['id'] in responses:
                    score = responses[q['id']]
                    weighted_score = (score - 1) / 4 * q['weight']  # 0-1に正規化

                    path = q['path']
                    pattern = q['pattern']

                    if path in path_scores:
                        if pattern in ['overuse', 'dysfunction', 'eruption', 'inferior_grip']:
                            path_scores[path][pattern] = path_scores[path].get(pattern, 0) + weighted_score
                        elif pattern in ['shadow_eruption', 'shadow_chaos', 'shadow_rigidity', 'shadow_obsession',
                                       'shadow_absorption', 'shadow_facade', 'shadow_rumination', 'shadow_hypochondria']:
                            # シャドウパターンの処理
                            path_scores[path]['shadow'] = path_scores[path].get('shadow', 0) + weighted_score
                        elif pattern in ['sleep_deprivation', 'sleep_quality', 'meal_irregularity', 'eating_disorder']:
                            # 身体的要因の処理
                            path_scores[path]['dysfunction'] = path_scores[path].get('dysfunction', 0) + weighted_score
                        elif pattern in ['isolation', 'somatic', 'dissociation', 'regression', 'rigidity']:
                            # その他のパターンも考慮
                            if path == 'general':
                                path_scores[path]['imbalance'] = path_scores[path].get('imbalance', 0) + weighted_score
                            else:
                                path_scores[path][pattern] = path_scores[path].get(pattern, 0) + weighted_score
                        path_scores[path]['total_weight'] += q['weight']

        # 正規化
        normalized_scores = {}
        for path, scores in path_scores.items():
            if scores['total_weight'] > 0:
                for pattern, value in scores.items():
                    if pattern != 'total_weight':
                        normalized_scores[f'{path}_{pattern}'] = value / scores['total_weight']

        # general_imbalanceの計算
        if 'general' in path_scores and path_scores['general']['total_weight'] > 0:
            normalized_scores['general_imbalance'] = path_scores['general']['imbalance'] / path_scores['general']['total_weight']

        return normalized_scores

    def apply_dysfunction_to_lambda3_state(self, state: AdvancedLambda3State,
                                         mbti_type: str,
                                         dysfunction_scores: Dict,
                                         responses: Dict[str, int] = None) -> None:
        """機能不全スコアをAdvancedLambda3Stateに適用"""

        profile = self.mbti_stacks[mbti_type]

        # 質問回答から自己側面を更新
        if responses:
            self.update_self_aspects_from_responses(state, responses, dysfunction_scores)

        # Ni過剰使用の影響
        if 'Ni_overuse' in dysfunction_scores:
            ni_overuse = dysfunction_scores['Ni_overuse']
            if ni_overuse > 0.7:
                state.path_preferences['Ni'] = PathPreference.DEPENDENT
                state.perception_noise += ni_overuse * 0.3
                # 理想と現実のギャップを拡大
                state.Λ_self_aspects['ideal'] = min(0.95, state.Λ_self_aspects['ideal'] + ni_overuse * 0.1)

        # Te機能不全の影響
        if 'Te_dysfunction' in dysfunction_scores:
            te_dysfunction = dysfunction_scores['Te_dysfunction']
            if te_dysfunction > 0.6:
                state.path_preferences['Te'] = PathPreference.RELUCTANT
                # 核となる自己の弱体化
                state.Λ_self_aspects['core'] *= (1 - te_dysfunction * 0.2)
                # 社会的自己も影響を受ける
                state.Λ_self_aspects['social'] *= (1 - te_dysfunction * 0.1)

        # Fi爆発の影響
        if 'Fi_eruption' in dysfunction_scores:
            fi_eruption = dysfunction_scores['Fi_eruption']
            if fi_eruption > 0.5:
                state.path_preferences['Fi'] = PathPreference.PREFERRED
                # 感情的な不安定さ
                state.emotional_state.anger += fi_eruption * 0.3
                state.emotional_state.sadness += fi_eruption * 0.2

        # Se劣等機能の暴走（最も重要）
        if 'Se_inferior_grip' in dysfunction_scores:
            se_grip = dysfunction_scores['Se_inferior_grip']
            if se_grip > 0.5:
                state.path_preferences['Se'] = PathPreference.AVOIDED
                # Shadowの急激な成長
                state.Λ_self_aspects['shadow'] = min(0.8,
                    state.Λ_self_aspects['shadow'] + se_grip * 0.3)
                # 全体のバランス崩壊
                state.σs *= (1 - se_grip * 0.3)
                state.ρT = min(0.95, state.ρT + se_grip * 0.4)
                # 身体感覚の低下
                state.Λ_bod *= (1 - se_grip * 0.2)

        # 全体的な不均衡
        if 'general_imbalance' in dysfunction_scores:
            imbalance = dysfunction_scores['general_imbalance']
            if imbalance > 0.6:
                # メタ認知の低下
                state.metacognition *= (1 - imbalance * 0.3)
                # トラウマカウントの増加
                for path in ['Ni', 'Te', 'Fi', 'Se']:
                    state.path_trauma_count[path] += int(imbalance * 5)

        # 発達段階の判定
        overall_dysfunction = np.mean(list(dysfunction_scores.values()))

        if overall_dysfunction > 0.7:
            state.developmental_stage = DevelopmentalStage.CRISIS
            print(f"  → 高い機能不全スコアにより CRISIS（危機期）と判定")
        elif overall_dysfunction > 0.5:
            if state.developmental_stage == DevelopmentalStage.STABILIZATION:
                state.developmental_stage = DevelopmentalStage.CRISIS
                print(f"  → 中程度の機能不全により安定期から危機期へ")

        # 経験タイプに基づく自己定義の更新
        state.update_self_definition('dysfunction', -overall_dysfunction)

        # シャドウ機能の影響を適用
        self.apply_shadow_effects(state, dysfunction_scores)

        # 睡眠・食事の影響を適用
        self.apply_physical_effects(state, responses, dysfunction_scores)

    def apply_shadow_effects(self, state: AdvancedLambda3State, dysfunction_scores: Dict) -> None:
        """シャドウ機能の影響をΛ³状態に適用"""

        # シャドウ機能の総合スコア
        shadow_scores = {
            'Ne': dysfunction_scores.get('Ne_shadow', 0),
            'Ti': dysfunction_scores.get('Ti_shadow', 0),
            'Fe': dysfunction_scores.get('Fe_shadow', 0),
            'Si': dysfunction_scores.get('Si_shadow', 0)
        }

        total_shadow_activation = sum(shadow_scores.values()) / 4

        # シャドウの活性化レベルに応じた影響
        if total_shadow_activation > 0.5:
            # 深刻なシャドウの噴出
            state.Λ_self_aspects['shadow'] = min(0.9,
                state.Λ_self_aspects['shadow'] + total_shadow_activation * 0.4)
            state.perception_noise += total_shadow_activation * 0.2

            # 各シャドウ機能の特異的影響
            if shadow_scores['Ne'] > 0.6:
                # Ne-shadow: 混沌とした可能性の追求
                state.path_preferences['Ne'] = PathPreference.DEPENDENT
                state.ρT = min(0.95, state.ρT + 0.3)  # テンション上昇

            if shadow_scores['Ti'] > 0.6:
                # Ti-shadow: 過度の内的論理への固執
                state.path_preferences['Ti'] = PathPreference.PREFERRED
                state.σs *= 0.7  # 社会的同期の低下

            if shadow_scores['Fe'] > 0.6:
                # Fe-shadow: 他者の感情への過剰同調
                state.path_preferences['Fe'] = PathPreference.DEPENDENT
                state.Λ_self_aspects['social'] *= 0.8

            if shadow_scores['Si'] > 0.6:
                # Si-shadow: 過去への病的な執着
                state.path_preferences['Si'] = PathPreference.PREFERRED
                state.Λ_bod *= 0.8  # 身体感覚の過敏化

    def apply_physical_effects(self, state: AdvancedLambda3State,
                             responses: Dict[str, int],
                             dysfunction_scores: Dict) -> None:
        """睡眠・食事の身体的要因をΛ³状態に適用"""

        if not responses:
            return

        # 睡眠の影響
        sleep_hours = responses.get('sleep_1', 3)
        sleep_quality = responses.get('sleep_2', 3)

        # 睡眠時間の影響（1:8h+ 2:7-8h 3:6-7h 4:5-6h 5:<5h）
        if sleep_hours >= 4:  # 6時間未満の睡眠
            sleep_deprivation_impact = (sleep_hours - 3) * 0.1  # 0.1-0.2
            state.Λ_bod *= (1 - sleep_deprivation_impact)
            state.ρT = min(0.95, state.ρT + sleep_deprivation_impact)
            state.perception_noise += sleep_deprivation_impact * 0.5

        # 睡眠の質の影響（1:とても良い 2:良い 3:普通 4:悪い 5:とても悪い）
        if sleep_quality >= 4:  # 睡眠の質が悪い
            sleep_quality_impact = (sleep_quality - 3) * 0.08  # 0.08-0.16
            state.Λ_bod *= (1 - sleep_quality_impact)
            state.metacognition *= (1 - sleep_quality_impact)
            state.emotional_state.anger += sleep_quality_impact * 0.3
            state.emotional_state.sadness += sleep_quality_impact * 0.2

        # 食事の影響
        meal_regularity = responses.get('meal_1', 3)
        meal_disorder = responses.get('meal_2', 3)

        # 食事の不規則性（1:常に規則的 2:ほぼ規則的 3:時々不規則 4:よく不規則 5:ほとんど不規則）
        if meal_regularity >= 4:  # 食事が不規則
            meal_impact = (meal_regularity - 3) * 0.06  # 0.06-0.12
            state.Λ_bod *= (1 - meal_impact)
            state.σs *= (1 - meal_impact * 0.5)
            state.current_desire.security -= meal_impact * 0.3

        # 食事異常（抜食・過食）
        if meal_disorder >= 4:  # 頻繁に食事を抜く/過食
            eating_disorder_impact = (meal_disorder - 3) * 0.08  # 0.08-0.16
            state.Λ_bod *= (1 - eating_disorder_impact)
            state.emotional_state.anger += eating_disorder_impact * 0.4
            state.current_desire.autonomy -= eating_disorder_impact * 0.2

            # 過食の場合は自己評価にも影響
            if meal_disorder == 5:  # 常に過食/抜食
                state.Λ_self_aspects['core'] *= (1 - eating_disorder_impact * 0.5)
                state.Λ_self_aspects['shadow'] += eating_disorder_impact * 0.3

        # 総合的な身体的ストレスの評価
        total_physical_stress = 0
        if sleep_hours >= 4:
            total_physical_stress += 0.3
        if sleep_quality >= 4:
            total_physical_stress += 0.2
        if meal_regularity >= 4:
            total_physical_stress += 0.2
        if meal_disorder >= 4:
            total_physical_stress += 0.3

        # 身体的ストレスが高い場合の追加影響
        if total_physical_stress >= 0.6:
            # 危機的な身体状態
            state.developmental_stage = DevelopmentalStage.CRISIS
            state.Λ_self_aspects['core'] *= 0.9
            state.perception_noise += 0.1
            print(f"  → 身体的ストレス蓄積により危機期へ移行")

        # 身体的要因によるPATH嗜好の変化
        if total_physical_stress >= 0.4:
            # 身体的疲労によりSeが回避される傾向
            state.path_preferences['Se'] = PathPreference.AVOIDED
            # 内向的機能への偏重
            state.path_preferences['Ni'] = PathPreference.DEPENDENT
            state.path_preferences['Fi'] = PathPreference.PREFERRED

    def update_self_aspects_from_responses(self, state: AdvancedLambda3State,
                                          responses: Dict[str, int],
                                          dysfunction_scores: Dict) -> None:
        """質問回答パターンから自己側面（social, ideal）を動的に更新"""

        # Social（社会的自己）への影響を計算
        social_impacts = {
            # Te機能不全は社会的効力感を低下
            'te_1': {'condition': responses.get('te_1', 0) >= 4, 'impact': -0.08},
            'te_2': {'condition': responses.get('te_2', 0) >= 3, 'impact': -0.05},

            # Fe-shadowは社会的仮面/混乱
            'sh_fe_1': {'condition': responses.get('sh_fe_1', 0) >= 3, 'impact': -0.06},
            'sh_fe_2': {'condition': responses.get('sh_fe_2', 0) >= 4, 'impact': -0.07},

            # 全般的機能低下
            'gen_2': {'condition': responses.get('gen_2', 0) >= 4, 'impact': -0.10},

            # Fi爆発は社会的関係を損なう
            'fi_1': {'condition': responses.get('fi_1', 0) >= 3, 'impact': -0.05}
        }

        # Ideal（理想自己）への影響を計算
        ideal_impacts = {
            # Ni過剰は理想を肥大化/歪曲
            'ni_1': {'condition': responses.get('ni_1', 0) >= 4, 'impact': -0.08},  # ネガティブな理想
            'ni_2': {'condition': responses.get('ni_2', 0) >= 5, 'impact': +0.10},  # 完璧主義的理想
            'ni_3': {'condition': responses.get('ni_3', 0) >= 4, 'impact': +0.05},  # 内的世界の理想化

            # Fi価値観の硬直化
            'fi_2': {'condition': responses.get('fi_2', 0) >= 3, 'impact': +0.06},

            # Shadow-Neの混沌
            'sh_ne_1': {'condition': responses.get('sh_ne_1', 0) >= 3, 'impact': -0.05},  # 理想の混乱

            # 自己喪失感
            'gen_1': {'condition': responses.get('gen_1', 0) >= 4, 'impact': -0.08}
        }

        # Socialの更新
        for key, rule in social_impacts.items():
            if rule['condition']:
                state.Λ_self_aspects['social'] = np.clip(
                    state.Λ_self_aspects['social'] + rule['impact'],
                    0.1, 1.0
                )

        # Idealの更新
        for key, rule in ideal_impacts.items():
            if rule['condition']:
                state.Λ_self_aspects['ideal'] = np.clip(
                    state.Λ_self_aspects['ideal'] + rule['impact'],
                    0.1, 1.0
                )

        # Core-Ideal-Socialの相互作用を反映
        self.apply_self_aspect_interactions(state, dysfunction_scores)

    def apply_self_aspect_interactions(self, state: AdvancedLambda3State,
                                      dysfunction_scores: Dict) -> None:
        """自己側面間の相互作用を適用"""

        # Ideal-Coreギャップの影響
        ideal_core_gap = state.Λ_self_aspects['ideal'] - state.Λ_self_aspects['core']

        if ideal_core_gap > 0.3:
            # 大きなギャップは社会的自己を不安定にする
            state.Λ_self_aspects['social'] *= 0.95
            # 認知的不協和も増加
            state.perception_noise += 0.02

        if ideal_core_gap < -0.2:
            # 理想が現実より低い（自己否定的）
            state.Λ_self_aspects['shadow'] += 0.05

        # Social-Coreギャップの影響
        social_core_gap = state.Λ_self_aspects['social'] - state.Λ_self_aspects['core']

        if social_core_gap > 0.2:
            # 社会的仮面の肥大（過剰適応）
            state.σs *= 0.9  # 真の同期率は低下

        if social_core_gap < -0.3:
            # 社会的引きこもり
            state.σs *= 0.8
            state.Λ_bod *= 0.95  # 身体的活力も低下

        # Shadow支配の影響
        if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
            # 影が核を上回る危機的状態
            state.Λ_self_aspects['social'] *= 0.9
            state.Λ_self_aspects['ideal'] *= 1.1  # 補償的に理想が肥大

def evaluate_depression_state(state: AdvancedLambda3State, dysfunction_scores: Dict) -> Dict:
    """より詳細なうつ状態評価"""

    # 基本指標
    indicators = {
        'energy_depletion': {
            'value': state.Λ_bod,
            'threshold': 0.5,  # 閾値を緩和
            'met': state.Λ_bod < 0.5
        },
        'self_worth_collapse': {
            'value': state.Λ_self_aspects['core'],
            'threshold': 0.7,  # 閾値を緩和
            'met': state.Λ_self_aspects['core'] < 0.7
        },
        'social_isolation': {
            'value': state.σs,
            'threshold': 0.4,  # 閾値を緩和
            'met': state.σs < 0.4
        },
        'cognitive_distortion': {
            'value': state.perception_noise,
            'threshold': 0.2,  # 閾値を緩和
            'met': state.perception_noise > 0.2
        },
        'shadow_dominance': {
            'value': state.Λ_self_aspects['shadow'] / state.Λ_self_aspects['core'],
            'threshold': 0.8,
            'met': state.Λ_self_aspects['shadow'] / state.Λ_self_aspects['core'] > 0.8
        },
        'metacognitive_impairment': {
            'value': state.metacognition,
            'threshold': 0.45,
            'met': state.metacognition < 0.45
        }
    }

    # MBTI特異的指標（INTJ）
    mbti_specific = {
        'ni_overuse': dysfunction_scores.get('Ni_overuse', 0) > 0.6,
        'se_grip': dysfunction_scores.get('Se_inferior_grip', 0) > 0.5,
        'te_dysfunction': dysfunction_scores.get('Te_dysfunction', 0) > 0.2,
        'general_imbalance': dysfunction_scores.get('general_imbalance', 0) > 0.7,
        # シャドウ機能の評価
        'shadow_activation': any([
            dysfunction_scores.get('Ne_shadow', 0) > 0.5,
            dysfunction_scores.get('Ti_shadow', 0) > 0.5,
            dysfunction_scores.get('Fe_shadow', 0) > 0.5,
            dysfunction_scores.get('Si_shadow', 0) > 0.5
        ])
    }

    # スコア計算
    basic_score = sum(1 for ind in indicators.values() if ind['met'])
    mbti_score = sum(1 for v in mbti_specific.values() if v)
    total_score = basic_score + mbti_score

    # 重症度判定（より細かい分類）
    if total_score >= 8:
        severity = "severe"
        interpretation = "重度のうつ状態・即座の介入が必要"
    elif total_score >= 6:
        severity = "moderate-severe"
        interpretation = "中等度から重度のうつ状態"
    elif total_score >= 4:
        severity = "moderate"
        interpretation = "中等度のうつ状態・機能不全が顕著"
    elif total_score >= 2:
        severity = "mild"
        interpretation = "軽度のうつ状態・早期介入推奨"
    else:
        severity = "subclinical"
        interpretation = "亜臨床的・予防的介入を検討"

    # 詳細な問題分析
    primary_issues = []
    if mbti_specific['ni_overuse']:
        primary_issues.append("Ni過剰使用による現実逃避")
    if mbti_specific['se_grip']:
        primary_issues.append("Se劣等機能の暴走")
    if indicators['metacognitive_impairment']['met']:
        primary_issues.append("メタ認知の著しい低下")
    if mbti_specific['general_imbalance']:
        primary_issues.append("全体的な心理的不均衡")
    if mbti_specific['shadow_activation']:
        primary_issues.append("シャドウ機能の活性化による人格の分裂傾向")

    return {
        'severity': severity,
        'interpretation': interpretation,
        'total_score': total_score,
        'basic_indicators': {k: v['met'] for k, v in indicators.items()},
        'mbti_indicators': mbti_specific,
        'primary_issues': primary_issues,
        'recommendations': generate_recommendations(severity, primary_issues)
    }

def generate_recommendations(severity: str, issues: List[str]) -> List[str]:
    """重症度と問題に基づく推奨事項"""
    recommendations = []

    if severity in ["severe", "moderate-severe"]:
        recommendations.append("専門家（精神科医・心理士）への相談を強く推奨")

    if "Ni過剰使用による現実逃避" in issues:
        recommendations.append("具体的な行動計画の作成と実行（小さなステップから）")
        recommendations.append("現実的な目標設定の練習")

    if "Se劣等機能の暴走" in issues:
        recommendations.append("健全な感覚体験の構造化（運動、自然散策など）")
        recommendations.append("衝動的行動の前に一時停止する練習")

    if "メタ認知の著しい低下" in issues:
        recommendations.append("日記やジャーナリングによる自己観察")
        recommendations.append("マインドフルネス練習")

    if "シャドウ機能の活性化による人格の分裂傾向" in issues:
        recommendations.append("シャドウワーク：抑圧された側面との対話")
        recommendations.append("夢分析やアクティブイマジネーション")
        recommendations.append("統合的な心理療法（ユング派分析など）の検討")

    # 身体的要因への対処
    if severity in ["moderate", "moderate-severe", "severe"]:
        recommendations.append("睡眠衛生の改善（7-8時間の確保）")
        recommendations.append("規則正しい食事リズムの確立")
        recommendations.append("血糖値の安定化（タンパク質を含む朝食）")

    return recommendations

def generate_emotional_interventions(state: AdvancedLambda3State,
                                   pathological: Dict) -> List[str]:
    """感情・欲求のバランス回復のための介入案"""

    interventions = []

    # 感情の抑圧への対処
    if any(ind['pattern'] == 'emotional_suppression'
           for ind in pathological['indicators']):
        interventions.append("感情日記：1日3回、感じた感情を記録する")
        interventions.append("感情の身体感覚に注目する練習")
        interventions.append("安全な環境での感情表現の練習")

    # 達成欲求と親和欲求の不均衡への対処
    if any(ind['pattern'] == 'achievement_isolation'
           for ind in pathological['indicators']):
        interventions.append("協働プロジェクトへの参加")
        interventions.append("成果よりもプロセスを楽しむ活動")
        interventions.append("親密な関係における脆弱性の練習")

    # 過度の独立性への対処
    if any(ind['pattern'] == 'hyper_independence'
           for ind in pathological['indicators']):
        interventions.append("小さな助けを求める練習")
        interventions.append("相互依存の健全性を学ぶ")
        interventions.append("チームワークを必要とする活動")

    # 感情状態に基づく具体的な提案
    if state.emotional_state.joy < 0.3:
        interventions.append("喜びの瞬間を意識的に味わう練習")

    if state.emotional_state.anger > 0.7:
        interventions.append("怒りの健全な表現方法の学習")

    return interventions

def enhanced_intj_assessment_with_emotions():
    """感情・欲求の歪みを含むINTJ評価（完全統合版）"""

    print("=== INTJ暴走状態評価システム（感情・欲求分析統合版） ===\n")

    # システム初期化
    assessor = MBTIDepressionAssessment()
    emotional_dynamics = MBTIEmotionalDynamics()

    # Lambda3状態の初期化
    lambda3_state = AdvancedLambda3State()

    # 初期の感情・欲求状態を表示
    print("初期状態:")
    print(f"  Core: {lambda3_state.Λ_self_aspects['core']:.2f}")
    print(f"  Ideal: {lambda3_state.Λ_self_aspects['ideal']:.2f}")
    print(f"  Social: {lambda3_state.Λ_self_aspects['social']:.2f}")
    print(f"  Shadow: {lambda3_state.Λ_self_aspects['shadow']:.2f}")
    print(f"  感情: Joy={lambda3_state.emotional_state.joy:.2f}, "
          f"Sadness={lambda3_state.emotional_state.sadness:.2f}, "
          f"Anger={lambda3_state.emotional_state.anger:.2f}")
    print(f"  欲求: Achievement={lambda3_state.current_desire.achievement:.2f}, "
          f"Affiliation={lambda3_state.current_desire.affiliation:.2f}, "
          f"Autonomy={lambda3_state.current_desire.autonomy:.2f}")

    # 質問への回答から機能不全スコアを計算
    sample_responses = {
        'ni_1': 4, 'ni_2': 5, 'ni_3': 4,
        'te_1': 4, 'te_2': 3,
        'fi_1': 2, 'fi_2': 3,
        'se_1': 5, 'se_2': 4, 'se_3': 3,
        'gen_1': 4, 'gen_2': 4,
        # シャドウ機能の回答
        'sh_ne_1': 3, 'sh_ne_2': 4,  # Ne-shadow
        'sh_ti_1': 4, 'sh_ti_2': 3,  # Ti-shadow
        'sh_fe_1': 3, 'sh_fe_2': 4,  # Fe-shadow
        'sh_si_1': 5, 'sh_si_2': 4,  # Si-shadow（過去への執着が強い）
        # 睡眠・食事
        'sleep_1': 4,  # 5-6時間睡眠
        'sleep_2': 4,  # 睡眠の質が悪い
        'meal_1': 3,   # 時々不規則
        'meal_2': 4    # 頻繁に食事を抜く/過食
    }

    dysfunction_scores = assessor.calculate_dysfunction_scores(sample_responses)

    # ストレスレベルを機能不全スコアから推定
    stress_level = np.mean(list(dysfunction_scores.values()))

    # MBTI特有の感情・欲求歪みを適用
    emotional_dynamics.apply_mbti_distortions(lambda3_state, 'INTJ', stress_level)

    # 機能不全を適用
    assessor.apply_dysfunction_to_lambda3_state(lambda3_state, 'INTJ', dysfunction_scores, sample_responses)

    print("\n歪み適用後の状態:")
    print(f"  Core: {lambda3_state.Λ_self_aspects['core']:.2f}")
    print(f"  Ideal: {lambda3_state.Λ_self_aspects['ideal']:.2f}")
    print(f"  Social: {lambda3_state.Λ_self_aspects['social']:.2f}")
    print(f"  Shadow: {lambda3_state.Λ_self_aspects['shadow']:.2f}")
    print(f"  感情: Joy={lambda3_state.emotional_state.joy:.2f}, "
          f"Sadness={lambda3_state.emotional_state.sadness:.2f}, "
          f"Anger={lambda3_state.emotional_state.anger:.2f}")
    print(f"  欲求: Achievement={lambda3_state.current_desire.achievement:.2f}, "
          f"Affiliation={lambda3_state.current_desire.affiliation:.2f}, "
          f"Autonomy={lambda3_state.current_desire.autonomy:.2f}")

    # 病理的パターンの検出
    pathological = emotional_dynamics.detect_pathological_patterns(lambda3_state, 'INTJ')

    print("\n=== 病理的パターン分析 ===")
    for indicator in pathological['indicators']:
        print(f"• {indicator['description']}")
        print(f"  深刻度: {indicator['severity']:.2f}")

    # 詳細な評価
    evaluation = evaluate_depression_state(lambda3_state, dysfunction_scores)

    print("\n=== 詳細評価結果 ===")
    print(f"うつ重症度: {evaluation['severity']}")
    print(f"解釈: {evaluation['interpretation']}")
    print(f"総合スコア: {evaluation['total_score']}/10")

    # 身体的要因の表示
    print("\n=== 身体的要因 ===")
    sleep_hours = sample_responses.get('sleep_1', 3)
    sleep_quality = sample_responses.get('sleep_2', 3)
    print(f"睡眠時間: {'8時間以上' if sleep_hours == 1 else '7-8時間' if sleep_hours == 2 else '6-7時間' if sleep_hours == 3 else '5-6時間' if sleep_hours == 4 else '5時間未満'}")
    print(f"睡眠の質: {'とても良い' if sleep_quality == 1 else '良い' if sleep_quality == 2 else '普通' if sleep_quality == 3 else '悪い' if sleep_quality == 4 else 'とても悪い'}")

    if sleep_hours >= 4:
        print("  → 睡眠不足による機能低下のリスク")
    if sleep_quality >= 4:
        print("  → 睡眠の質の低下による回復不足")

    print("\n基本指標:")
    for key, met in evaluation['basic_indicators'].items():
        status = "✗" if met else "✓"
        print(f"  {status} {key}")

    print("\nMBTI特異的指標:")
    for key, met in evaluation['mbti_indicators'].items():
        status = "✗" if met else "✓"
        print(f"  {status} {key}")

    print("\n主要な問題:")
    for issue in evaluation['primary_issues']:
        print(f"  • {issue}")

    # 統合的な介入提案
    print("\n=== 統合的介入提案 ===")

    # 機能不全に基づく推奨事項
    print("\n【機能回復のための介入】")
    for rec in evaluation['recommendations']:
        print(f"  • {rec}")

    # 感情・欲求バランスのための介入
    emotional_interventions = generate_emotional_interventions(lambda3_state, pathological)
    print("\n【感情・欲求バランスのための介入】")
    for intervention in emotional_interventions:
        print(f"  • {intervention}")

    return lambda3_state, evaluation

# 実行例
if __name__ == "__main__":
    result_state, evaluation = enhanced_intj_assessment_with_emotions()
