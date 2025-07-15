import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import random
from enum import Enum
import networkx as nx

# ユング8機能ベースPATH
BASE_PATHS = ['Ne', 'Ni', 'Te', 'Ti', 'Fe', 'Fi', 'Se', 'Si']

# 基本的なデータクラス定義
@dataclass
class Context:
    """詳細なコンテキスト定義"""
    category: str  # 'work', 'relationship', 'hobby', 'crisis'
    social_distance: float  # 0.0(親密) ~ 1.0(他人)
    formality: float  # 0.0(カジュアル) ~ 1.0(フォーマル)
    stress_level: float  # 0.0(リラックス) ~ 1.0(高ストレス)

    def to_vector(self):
        return np.array([self.social_distance, self.formality, self.stress_level])

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


# PersonalitySimulator基本クラス
class PersonalitySimulator:
    def __init__(self, base_scores: Dict[str, float]):
        self.base_scores = base_scores
        self.state = Lambda3State()
        self.history = []

    def simulate_transaction(self, context: Context) -> Dict:
        """単一トランザクションのシミュレート"""
        # 簡易版の実装
        probs = self.calculate_path_probabilities(context)
        paths, weights = zip(*probs.items())
        chosen_path = random.choices(paths, weights=weights, k=1)[0]

        outcome = random.gauss(0, 0.5)
        self.state.update_learning(chosen_path, outcome)

        record = {
            't': len(self.history),
            'context': context,
            'path': chosen_path,
            'probs': probs,
            'outcome': outcome
        }
        self.history.append(record)
        return record

    def calculate_path_probabilities(self, context: Context) -> Dict[str, float]:
        """PATH確率計算（簡易版）"""
        probs = {}
        for path in BASE_PATHS:
            score = self.base_scores[path]
            # コンテキストによる簡易修正
            if context.stress_level > 0.7 and path.endswith('i'):
                score *= 1.2
            probs[path] = max(0, score)

        # 正規化
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        return probs

class PersonalityNetwork:
    """複数の人格モデルのネットワーク"""
    def __init__(self):
        self.personalities: Dict[str, EnhancedPersonalitySimulator] = {}
        self.interaction_network = nx.Graph()

    def add_personality(self, person_id: str, base_scores: Dict[str, float]):
        """新しい人格を追加"""
        self.personalities[person_id] = EnhancedPersonalitySimulator(base_scores)
        self.personalities[person_id].person_id = person_id  # person_idを後から設定
        self.interaction_network.add_node(person_id)

    def add_relationship(self, person1: str, person2: str, strength: float):
        """関係性を追加"""
        self.interaction_network.add_edge(
            person1, person2, weight=strength
        )

    def simulate_interaction(self, person1_id: str, person2_id: str,
                           context: Context) -> Dict:
        """2者間の相互作用をシミュレート"""
        p1 = self.personalities[person1_id]
        p2 = self.personalities[person2_id]

        # 両者のPATH選択
        p1_path = p1.simulate_transaction(context)['path']
        p2_path = p2.simulate_transaction(context)['path']

        # 相互フィードバック生成
        compatibility = self._calculate_path_compatibility(p1_path, p2_path)

        # フィードバックを交換
        p1_feedback = SocialFeedback(
            source_id=person2_id,
            valence=compatibility,
            intensity=0.7,
            authenticity=0.8,
            category='interaction'
        )
        p2_feedback = SocialFeedback(
            source_id=person1_id,
            valence=compatibility,
            intensity=0.7,
            authenticity=0.8,
            category='interaction'
        )

        p1.state.process_social_feedback(p2_feedback)
        p2.state.process_social_feedback(p1_feedback)

        # 関係性の更新
        edge_data = self.interaction_network.get_edge_data(person1_id, person2_id)
        if edge_data:
            new_weight = edge_data['weight'] * 0.9 + compatibility * 0.1
            self.interaction_network[person1_id][person2_id]['weight'] = new_weight

        return {
            'person1': {'id': person1_id, 'path': p1_path},
            'person2': {'id': person2_id, 'path': p2_path},
            'compatibility': compatibility
        }

    def _calculate_path_compatibility(self, path1: str, path2: str) -> float:
        """PATH間の相性を計算（完全版）"""

        # 基本的な相性マトリックス
        base_compatibility = {
            # 同じ機能同士
            ('Ne', 'Ne'): 0.7, ('Ni', 'Ni'): 0.5, ('Te', 'Te'): 0.4, ('Ti', 'Ti'): 0.6,
            ('Fe', 'Fe'): 0.8, ('Fi', 'Fi'): 0.5, ('Se', 'Se'): 0.7, ('Si', 'Si'): 0.6,

            # Ne との組み合わせ
            ('Ne', 'Ni'): 0.6, ('Ne', 'Te'): 0.8, ('Ne', 'Ti'): 0.7, ('Ne', 'Fe'): 0.9,
            ('Ne', 'Fi'): 0.6, ('Ne', 'Se'): 0.8, ('Ne', 'Si'): 0.3,

            # Ni との組み合わせ
            ('Ni', 'Te'): 0.8, ('Ni', 'Ti'): 0.7, ('Ni', 'Fe'): 0.7, ('Ni', 'Fi'): 0.8,
            ('Ni', 'Se'): 0.5, ('Ni', 'Si'): 0.4,

            # Te との組み合わせ
            ('Te', 'Ti'): 0.5, ('Te', 'Fe'): 0.4, ('Te', 'Fi'): 0.4,
            ('Te', 'Se'): 0.8, ('Te', 'Si'): 0.7,

            # Ti との組み合わせ
            ('Ti', 'Fe'): 0.3, ('Ti', 'Fi'): 0.6, ('Ti', 'Se'): 0.5, ('Ti', 'Si'): 0.7,

            # Fe との組み合わせ
            ('Fe', 'Fi'): 0.5, ('Fe', 'Se'): 0.8, ('Fe', 'Si'): 0.7,

            # Fi との組み合わせ
            ('Fi', 'Se'): 0.6, ('Fi', 'Si'): 0.7,

            # Se と Si
            ('Se', 'Si'): 0.4,
        }

        # キーを正規化
        key = tuple(sorted([path1, path2]))

        # 基本相性を取得
        return base_compatibility.get(key, 0.5)

class EnhancedPersonalitySimulator(PersonalitySimulator):
    """長期発達対応の拡張シミュレーター"""
    def __init__(self, base_scores: Dict[str, float]):
        super().__init__(base_scores)
        self.person_id = "unknown"  # デフォルト値
        self.state = AdvancedLambda3State()
        self.life_events = []

    def apply_random_event(self):
        """ランダムな出来事を適用"""
        event_types = [
            ('loss', -0.8, 0.3),      # 喪失
            ('opportunity', 0.7, 0.5), # 機会
            ('conflict', -0.5, 0.6),   # 葛藤
            ('achievement', 0.8, 0.2), # 達成
            ('failure', -0.6, 0.4),    # 失敗
            ('insight', 0.6, 0.1),     # 洞察（メタ認知向上）
            ('trauma', -0.9, 0.8),     # トラウマ
        ]

        if random.random() < 0.2:  # 20%の確率でイベント
            event_type, impact, uncertainty = random.choice(event_types)
            event = Event(
                event_type=event_type,
                impact=impact,
                uncertainty=uncertainty,
                duration=random.randint(1, 5)
            )
            self.life_events.append(event)

            # イベントの影響を適用
            self._process_life_event(event)

            # 洞察イベントでメタ認知向上
            if event_type == 'insight':
                self.state.metacognition = min(1.0,
                    self.state.metacognition + 0.1)

    def _process_life_event(self, event: Event):
        """人生の出来事を処理"""
        # 認識ノイズの増加
        self.state.perception_noise += event.uncertainty * 0.1

        # 感情への影響
        if event.impact < -0.5:
            self.state.emotional_state.sadness += abs(event.impact) * 0.3
            self.state.emotional_state.fear += event.uncertainty * 0.2
        elif event.impact > 0.5:
            self.state.emotional_state.joy += event.impact * 0.3

        # 自己定義への影響
        self.state.update_self_definition(event.event_type, event.impact)

    def simulate_long_term(self, num_transactions: int):
        """長期シミュレーション"""
        contexts = [
            Context('work', 0.6, 0.8, 0.7),
            Context('relationship', 0.2, 0.1, 0.3),
            Context('hobby', 0.4, 0.2, 0.2),
            Context('crisis', 0.8, 0.5, 0.9),
        ]

        for t in range(num_transactions):
            # ランダムイベントのチェック
            self.apply_random_event()

            # 通常のトランザクション
            ctx = random.choice(contexts)

            # 認識ノイズの適用
            if self.state.perception_noise > 0:
                # コンテキストの誤認識
                ctx.stress_level += random.gauss(0, self.state.perception_noise)
                ctx.stress_level = np.clip(ctx.stress_level, 0, 1)

            result = self.simulate_transaction(ctx)

            # 長期的なPATH嗜好の更新
            self._update_path_preferences(result['path'], result['outcome'])

            # メタ認知の成長（学習による）
            if result['outcome'] > 0.5:
                self.state.metacognition = min(1.0,
                    self.state.metacognition + 0.01)

            # 発達段階のチェック
            self.state.check_developmental_transition()

            # 定期的な状態出力
            if t % 10 == 0:
                self._print_developmental_status(t)

    def _update_path_preferences(self, path: str, outcome: float):
        """長期的なPATH嗜好の更新"""
        current_pref = self.state.path_preferences[path]

        if outcome < -0.5:
            self.state.path_trauma_count[path] += 1

            # トラウマの蓄積による嗜好変化
            if self.state.path_trauma_count[path] > 3:
                if current_pref != PathPreference.AVOIDED:
                    self.state.path_preferences[path] = PathPreference.RELUCTANT
                if self.state.path_trauma_count[path] > 5:
                    self.state.path_preferences[path] = PathPreference.AVOIDED

        elif outcome > 0.5:
            # ポジティブな経験による強化
            if current_pref == PathPreference.NEUTRAL:
                self.state.path_preferences[path] = PathPreference.PREFERRED
            elif current_pref == PathPreference.PREFERRED:
                # 過度の依存チェック
                preferred_count = sum(1 for p in self.state.path_preferences.values()
                                    if p == PathPreference.PREFERRED)
                if preferred_count < 3:  # 偏りすぎ防止
                    self.state.path_preferences[path] = PathPreference.DEPENDENT

    def _print_developmental_status(self, t: int):
        """発達状態の出力"""
        print(f"\n[{self.person_id}] Transaction {t}")
        print(f"発達段階: {self.state.developmental_stage.value}")
        print(f"自己定義: Core={self.state.Λ_self_aspects['core']:.2f}, "
              f"Shadow={self.state.Λ_self_aspects['shadow']:.2f}")
        print(f"メタ認知: {self.state.metacognition:.2f}")

        # PATH嗜好の要約
        avoided = [p for p, pref in self.state.path_preferences.items()
                  if pref == PathPreference.AVOIDED]
        preferred = [p for p, pref in self.state.path_preferences.items()
                    if pref == PathPreference.PREFERRED]
        print(f"回避PATH: {avoided}, 好むPATH: {preferred}")

# 実行例：集団シミュレーション
def run_group_simulation():
    # 人格ネットワークの作成
    network = PersonalityNetwork()

    # 異なる性格タイプの個体を追加
    personalities = {
        'person_A': {'Ne': 15, 'Ni': 5, 'Te': 12, 'Ti': 6,
                     'Fe': 14, 'Fi': 4, 'Se': 8, 'Si': 6},  # 外向的
        'person_B': {'Ne': 6, 'Ni': 14, 'Te': 7, 'Ti': 13,
                     'Fe': 5, 'Fi': 12, 'Se': 6, 'Si': 10},  # 内向的
        'person_C': {'Ne': 10, 'Ni': 10, 'Te': 10, 'Ti': 10,
                     'Fe': 10, 'Fi': 10, 'Se': 10, 'Si': 10}  # バランス型
    }

    for person_id, scores in personalities.items():
        network.add_personality(person_id, scores)

    # 関係性の設定
    network.add_relationship('person_A', 'person_B', 0.3)
    network.add_relationship('person_A', 'person_C', 0.6)
    network.add_relationship('person_B', 'person_C', 0.5)

    # 長期シミュレーション with 相互作用
    for t in range(100):
        # 個別の長期発達
        for person_id, sim in network.personalities.items():
            sim.apply_random_event()

        # 定期的な相互作用
        if t % 5 == 0:
            # ランダムなペアで相互作用
            persons = list(network.personalities.keys())
            if len(persons) >= 2:
                p1, p2 = random.sample(persons, 2)
                if network.interaction_network.has_edge(p1, p2):
                    ctx = Context('relationship', 0.3, 0.2, 0.4)
                    result = network.simulate_interaction(p1, p2, ctx)
                    print(f"\n[t={t}] {p1} ↔ {p2}: "
                          f"相性={result['compatibility']:.2f}")

    # 最終的なネットワーク状態の可視化
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(network.interaction_network)

    # エッジの重みで太さを変える
    edges = network.interaction_network.edges()
    weights = [network.interaction_network[u][v]['weight']
              for u, v in edges]

    nx.draw(network.interaction_network, pos,
            width=[w*5 for w in weights],
            node_color='lightblue',
            node_size=3000,
            with_labels=True,
            font_size=10)

    plt.title("Personality Network After Long-term Interaction")
    plt.show()

if __name__ == "__main__":
    # 個人の長期発達シミュレーション
    print("=== 個人の長期発達シミュレーション ===")
    sim = EnhancedPersonalitySimulator(
        {'Ne': 12, 'Ni': 8, 'Te': 15, 'Ti': 6,
         'Fe': 9, 'Fi': 7, 'Se': 10, 'Si': 5}
    )
    sim.person_id = "test_person"  # person_idを後から設定
    sim.simulate_long_term(100)

    # 集団シミュレーション
    print("\n\n=== 集団相互作用シミュレーション ===")
    run_group_simulation()
