import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

# =========== 既存のコードから必要な部分をインポート ===========

# ユング8機能ベースPATH
BASE_PATHS = ['Ne', 'Ni', 'Te', 'Ti', 'Fe', 'Fi', 'Se', 'Si']

@dataclass
class MBTIProfile:
    """MBTIプロファイル"""
    type_code: str
    dominant: str
    auxiliary: str
    tertiary: str
    inferior: str

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
    achievement: float = 0.5
    affiliation: float = 0.5
    power: float = 0.5
    security: float = 0.5
    autonomy: float = 0.5

    def to_vector(self):
        return np.array([self.achievement, self.affiliation,
                        self.power, self.security, self.autonomy])

# 発達段階の定義
class DevelopmentalStage(Enum):
    EXPLORATION = "exploration"
    FORMATION = "formation"
    STABILIZATION = "stabilization"
    INTEGRATION = "integration"
    CRISIS = "crisis"

# PATH嗜好の状態
class PathPreference(Enum):
    AVOIDED = "avoided"
    RELUCTANT = "reluctant"
    NEUTRAL = "neutral"
    PREFERRED = "preferred"
    DEPENDENT = "dependent"

@dataclass
class SocialFeedback:
    """社会的フィードバック"""
    source_id: str
    valence: float
    intensity: float
    authenticity: float
    category: str

@dataclass
class Event:
    """不確実な出来事"""
    event_type: str
    impact: float
    uncertainty: float
    duration: int

# =========== 新規追加：高度分析機能 ===========

@dataclass
class IdealMBTIProfile:
    """理想的なMBTIプロファイル"""
    mbti_type: str
    ideal_function_levels: Dict[str, float]
    ideal_function_balance: Dict[str, float]
    ideal_emotional_state: Dict[str, float]
    ideal_desire_state: Dict[str, float]
    ideal_self_aspects: Dict[str, float]
    ideal_basic_tensors: Dict[str, float]

@dataclass
class ManifoldPoint:
    """Λ³マニフォールド上の点"""
    coordinates: np.ndarray
    curvature: float
    nearby_attractors: List[str]

@dataclass
class AttractorBasin:
    """アトラクター盆地"""
    name: str
    center: np.ndarray
    radius: float
    basin_depth: float
    stability_index: float
    attractor_type: str  # 'healthy' or 'pathological'

@dataclass
class DeviationQuality:
    """乖離の質的分類"""
    deviation_type: str  # 'constructive' or 'destructive'
    pattern: str
    magnitude: float
    trajectory: str
    growth_potential: float

class Lambda3ManifoldAnalyzer:
    """Λ³マニフォールド解析器"""
    
    def __init__(self):
        self.metric_tensor = self._initialize_metric_tensor()
    
    def _initialize_metric_tensor(self):
        """Riemann計量テンソルの初期化"""
        # 簡略化されたメトリックテンソル（8x8 for BASE_PATHS）
        g = np.eye(8)
        
        # 主機能間の関係による空間の歪み
        # Ni-Te (INTJ主-補助) は近い
        g[1, 2] = g[2, 1] = 0.8
        # Ni-Se (主-劣等) は遠い
        g[1, 6] = g[6, 1] = 0.2
        # Shadow機能は通常機能から遠い
        for i in range(4):
            for j in range(4, 8):
                g[i, j] = g[j, i] = 0.3
        
        return g
    
    def analyze(self, state: 'DynamicLambda3State', 
                dysfunction_scores: Dict[str, float]) -> Dict:
        """マニフォールド解析の実行"""
        
        # 現在状態のマニフォールド座標を計算
        current_point = self._calculate_manifold_coordinates(state)
        
        # 局所的な曲率を計算（Shadow活性化時は空間が歪む）
        curvature = self._calculate_local_curvature(state, dysfunction_scores)
        
        # 近傍のアトラクターを特定
        nearby_attractors = self._identify_nearby_attractors(current_point)
        
        # 健全状態への測地線距離
        healthy_state = self._define_healthy_state_coordinates()
        geodesic_distance = self._calculate_geodesic_distance(
            current_point, healthy_state, curvature
        )
        
        # 位相的特徴
        topological_features = self._analyze_topological_features(
            state, dysfunction_scores
        )
        
        return {
            'current_position': ManifoldPoint(
                coordinates=current_point,
                curvature=curvature,
                nearby_attractors=nearby_attractors
            ),
            'geodesic_distance_to_health': geodesic_distance,
            'manifold_dimension': len(current_point),
            'topological_features': topological_features,
            'phase_space_velocity': self._calculate_phase_velocity(state),
            'critical_points': self._identify_critical_points(state, dysfunction_scores)
        }
    
    def _calculate_manifold_coordinates(self, state) -> np.ndarray:
        """状態のマニフォールド座標を計算"""
        coords = []
        
        # PATH活性度を基本座標として使用
        for path in BASE_PATHS:
            coords.append(state.path_states.get(path, 0.5))
        
        # 高次元への埋め込み（感情・欲求次元）
        coords.extend([
            state.emotional_state.joy,
            state.emotional_state.sadness,
            state.emotional_state.anger,
            state.current_desire.achievement,
            state.current_desire.autonomy
        ])
        
        return np.array(coords)
    
    def _calculate_local_curvature(self, state, dysfunction_scores) -> float:
        """局所的な空間の曲率を計算"""
        # Shadow活性化による空間の歪み
        shadow_activation = sum([
            dysfunction_scores.get(f'{path}_shadow', 0) 
            for path in ['Ne', 'Ti', 'Fe', 'Si']
        ]) / 4
        
        # 機能間葛藤による曲率増加
        conflict_curvature = 0.0
        if hasattr(state, 'path_interaction'):
            for (p1, p2), strength in state.path_interaction.resonance_factors.items():
                if strength < -0.5:  # 強い負の相関
                    conflict_curvature += abs(strength) * 0.1
        
        # 基底曲率 + Shadow歪み + 葛藤曲率
        return 0.1 + shadow_activation * 0.5 + conflict_curvature
    
    def _calculate_geodesic_distance(self, point1: np.ndarray, 
                                   point2: np.ndarray, 
                                   curvature: float) -> float:
        """測地線距離の計算（簡略化版）"""
        # ユークリッド距離
        euclidean = np.linalg.norm(point2 - point1)
        
        # 曲率による補正（曲率が高いほど実際の距離は長い）
        geodesic = euclidean * (1 + curvature)
        
        # メトリックテンソルによる重み付け
        diff = point2[:8] - point1[:8]  # PATH次元のみ
        weighted_dist = np.sqrt(diff.T @ self.metric_tensor @ diff)
        
        return (geodesic + weighted_dist) / 2
    
    def _define_healthy_state_coordinates(self) -> np.ndarray:
        """健全状態の座標を定義（INTJ用）"""
        coords = [
            0.2,  # Ne (低)
            0.8,  # Ni (高)
            0.7,  # Te (高)
            0.2,  # Ti (低)
            0.2,  # Fe (低)
            0.5,  # Fi (中)
            0.4,  # Se (低～中)
            0.2,  # Si (低)
            # 感情・欲求
            0.6,  # joy
            0.2,  # sadness
            0.2,  # anger
            0.7,  # achievement
            0.8   # autonomy
        ]
        return np.array(coords)
    
    def _identify_nearby_attractors(self, point: np.ndarray) -> List[str]:
        """近傍のアトラクターを特定"""
        nearby = []
        
        # 簡略化：PATH活性度パターンから判定
        ni_dominant = point[1] > 0.7
        shadow_active = sum(point[4:8]) > 1.5
        
        if ni_dominant and not shadow_active:
            nearby.append("healthy_integration")
        if shadow_active:
            nearby.append("shadow_grip")
        if point[6] > 0.6:  # Se
            nearby.append("inferior_eruption")
            
        return nearby
    
    def _analyze_topological_features(self, state, dysfunction_scores) -> Dict:
        """位相的特徴の分析"""
        return {
            'connectivity': self._calculate_network_connectivity(state),
            'holes': self._detect_topological_holes(state),
            'dimension_reduction': self._check_dimension_collapse(state)
        }
    
    def _calculate_network_connectivity(self, state) -> float:
        """ネットワーク連結性"""
        # PATH間の正の相互作用の数
        positive_connections = 0
        if hasattr(state, 'path_interaction'):
            for strength in state.path_interaction.resonance_factors.values():
                if strength > 0.3:
                    positive_connections += 1
        
        return positive_connections / 28  # 8C2 = 28
    
    def _detect_topological_holes(self, state) -> List[str]:
        """位相的な穴（断絶）の検出"""
        holes = []
        
        # 自己側面間の大きなギャップ
        if abs(state.Λ_self_aspects['ideal'] - state.Λ_self_aspects['core']) > 0.5:
            holes.append("ideal_reality_gap")
        
        if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
            holes.append("shadow_dominance_hole")
            
        return holes
    
    def _check_dimension_collapse(self, state) -> bool:
        """次元崩壊のチェック（多様性の喪失）"""
        # すべてのPATHが似た値になっているか
        path_values = [state.path_states[p] for p in BASE_PATHS]
        variance = np.var(path_values)
        return variance < 0.05
    
    def _calculate_phase_velocity(self, state) -> float:
        """位相空間での速度（変化の速さ）"""
        # 簡略化：最近の拍動頻度から推定
        if hasattr(state, 'pulsation_history') and state.pulsation_history:
            recent_pulses = len([p for p in state.pulsation_history[-10:]])
            return recent_pulses / 10.0
        return 0.1
    
    def _identify_critical_points(self, state, dysfunction_scores) -> List[Dict]:
        """臨界点の特定"""
        critical_points = []
        
        # Shadowの臨界的活性化
        shadow_total = sum(dysfunction_scores.get(f'{p}_shadow', 0) 
                          for p in ['Ne', 'Ti', 'Fe', 'Si'])
        if shadow_total > 2.0:
            critical_points.append({
                'type': 'shadow_critical',
                'severity': shadow_total / 4,
                'description': 'Shadow機能群の臨界的活性化'
            })
        
        # 自己崩壊の臨界点
        if state.Λ_self_aspects['core'] < 0.3:
            critical_points.append({
                'type': 'self_collapse',
                'severity': 1.0 - state.Λ_self_aspects['core'],
                'description': 'Core自己の崩壊危機'
            })
            
        return critical_points

class AttractorBasinAnalyzer:
    """アトラクター盆地解析器"""
    
    def __init__(self, mbti_type: str):
        self.mbti_type = mbti_type
        self.attractors = self._initialize_attractors()
    
    def _initialize_attractors(self) -> List[AttractorBasin]:
        """アトラクターの初期化"""
        if self.mbti_type == 'INTJ':
            return [
                # 健全なアトラクター
                AttractorBasin(
                    name="balanced_integration",
                    center=np.array([0.8, 0.7, 0.5, 0.4, 0.2, 0.2, 0.2, 0.2]),
                    radius=0.3,
                    basin_depth=0.8,
                    stability_index=0.9,
                    attractor_type="healthy"
                ),
                AttractorBasin(
                    name="creative_tension",
                    center=np.array([0.9, 0.6, 0.6, 0.3, 0.3, 0.3, 0.2, 0.2]),
                    radius=0.25,
                    basin_depth=0.6,
                    stability_index=0.7,
                    attractor_type="healthy"
                ),
                # 病理的アトラクター
                AttractorBasin(
                    name="ni_ti_loop",
                    center=np.array([0.9, 0.2, 0.2, 0.9, 0.1, 0.1, 0.1, 0.1]),
                    radius=0.2,
                    basin_depth=0.9,
                    stability_index=0.95,  # 抜け出しにくい
                    attractor_type="pathological"
                ),
                AttractorBasin(
                    name="shadow_chaos",
                    center=np.array([0.7, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7, 0.7]),
                    radius=0.35,
                    basin_depth=0.7,
                    stability_index=0.5,  # 不安定
                    attractor_type="pathological"
                )
            ]
        return []
    
    def analyze(self, state: 'DynamicLambda3State') -> Dict:
        """アトラクター解析の実行"""
        
        # 現在状態のベクトル表現
        current_vector = self._state_to_vector(state)
        
        # 各アトラクターへの距離と引力
        attractor_forces = []
        for attractor in self.attractors:
            distance = np.linalg.norm(current_vector - attractor.center)
            if distance < attractor.radius * 2:  # 影響圏内
                force = attractor.basin_depth / (distance + 0.1)
                attractor_forces.append({
                    'attractor': attractor,
                    'distance': distance,
                    'force': force,
                    'in_basin': distance < attractor.radius
                })
        
        # 最も近い/強いアトラクター
        if attractor_forces:
            nearest = min(attractor_forces, key=lambda x: x['distance'])
            strongest = max(attractor_forces, key=lambda x: x['force'])
        else:
            nearest = strongest = None
        
        # 軌道予測
        trajectory = self._predict_trajectory(current_vector, attractor_forces)
        
        # 安定性分析
        stability = self._analyze_stability(state, attractor_forces)
        
        return {
            'nearby_attractors': attractor_forces,
            'nearest_attractor': nearest,
            'strongest_attractor': strongest,
            'trajectory_prediction': trajectory,
            'stability_analysis': stability,
            'escape_difficulty': self._calculate_escape_difficulty(nearest),
            'phase_portrait': self._generate_phase_portrait(state)
        }
    
    def _state_to_vector(self, state) -> np.ndarray:
        """状態をベクトルに変換"""
        return np.array([
            state.path_states.get('Ni', 0.5),
            state.path_states.get('Te', 0.5),
            state.path_states.get('Fi', 0.5),
            state.path_states.get('Se', 0.5),
            state.path_states.get('Ne', 0.1),
            state.path_states.get('Ti', 0.1),
            state.path_states.get('Fe', 0.1),
            state.path_states.get('Si', 0.1)
        ])
    
    def _predict_trajectory(self, current: np.ndarray, 
                          forces: List[Dict]) -> Dict:
        """軌道予測"""
        if not forces:
            return {
                'direction': 'drift', 
                'target': None, 
                'target_type': None,  # 追加
                'estimated_time': 0,
                'confidence': 0.1
            }
        
        # 最も強い引力の方向
        strongest_force = max(forces, key=lambda x: x['force'])
        direction_vector = strongest_force['attractor'].center - current
        direction_vector = direction_vector / (np.linalg.norm(direction_vector) + 0.01)
        
        return {
            'direction': 'attracted',
            'target': strongest_force['attractor'].name,
            'target_type': strongest_force['attractor'].attractor_type,
            'estimated_time': strongest_force['distance'] / 0.1,  # 簡略化
            'confidence': min(strongest_force['force'] / 10, 1.0)
        }
    
    def _analyze_stability(self, state, forces: List[Dict]) -> Dict:
        """安定性分析"""
        # 複数のアトラクターからの引力がある場合は不安定
        if len([f for f in forces if f['force'] > 1.0]) > 1:
            stability_type = 'bistable'
            stability_index = 0.3
        elif forces and forces[0]['in_basin']:
            stability_type = 'stable'
            stability_index = forces[0]['attractor'].stability_index
        else:
            stability_type = 'unstable'
            stability_index = 0.1
        
        return {
            'type': stability_type,
            'index': stability_index,
            'oscillation_risk': self._calculate_oscillation_risk(forces)
        }
    
    def _calculate_oscillation_risk(self, forces: List[Dict]) -> float:
        """振動リスク（複数アトラクター間の行き来）"""
        if len(forces) < 2:
            return 0.0
        
        # 上位2つのアトラクターの力が拮抗している場合
        sorted_forces = sorted(forces, key=lambda x: x['force'], reverse=True)
        if sorted_forces[0]['force'] / (sorted_forces[1]['force'] + 0.01) < 1.5:
            return 0.8
        return 0.2
    
    def _calculate_escape_difficulty(self, nearest: Optional[Dict]) -> float:
        """脱出難易度"""
        if not nearest or not nearest['in_basin']:
            return 0.0
        
        attractor = nearest['attractor']
        # 深さ × 安定性 × (1 - 健全度)
        if attractor.attractor_type == 'pathological':
            return attractor.basin_depth * attractor.stability_index
        return 0.0  # 健全なアトラクターからの脱出は問題ない
    
    def _generate_phase_portrait(self, state) -> Dict:
        """位相図の生成（簡略版）"""
        return {
            'current_phase': self._determine_phase(state),
            'velocity_field': self._estimate_velocity_field(state)
        }
    
    def _determine_phase(self, state) -> str:
        """現在の位相を判定"""
        if state.developmental_stage == DevelopmentalStage.CRISIS:
            return 'critical_transition'
        elif state.ρT > 0.8:
            return 'high_tension'
        elif state.metacognition < 0.3:
            return 'low_awareness'
        else:
            return 'normal_dynamics'
    
    def _estimate_velocity_field(self, state) -> np.ndarray:
        """速度場の推定"""
        # 簡略化：主要な変化の方向
        return np.array([
            state.perception_noise * 0.1,  # ノイズによる撹乱
            -state.metacognition * 0.05,   # メタ認知の影響
            state.ρT - 0.5                 # テンション駆動
        ])

class DeviationQualityAnalyzer:
    """乖離の質的分析器"""
    
    def analyze(self, state: 'DynamicLambda3State',
                manifold_result: Dict,
                attractor_result: Dict) -> Dict:
        """乖離の質的分析を実行"""
        
        # 乖離パターンの分類
        deviation_pattern = self._classify_deviation_pattern(
            state, manifold_result, attractor_result
        )
        
        # 成長可能性の評価
        growth_potential = self._evaluate_growth_potential(
            state, deviation_pattern, attractor_result
        )
        
        # 建設的/破壊的の判定
        quality_type = self._determine_deviation_quality(
            deviation_pattern, growth_potential, state
        )
        
        # 推奨される対応
        recommendations = self._generate_quality_based_recommendations(
            quality_type, deviation_pattern
        )
        
        return {
            'deviation_quality': DeviationQuality(
                deviation_type=quality_type,
                pattern=deviation_pattern,
                magnitude=manifold_result['geodesic_distance_to_health'],
                trajectory=attractor_result['trajectory_prediction']['direction'],
                growth_potential=growth_potential
            ),
            'constructive_aspects': self._identify_constructive_aspects(state),
            'destructive_aspects': self._identify_destructive_aspects(state),
            'transformation_readiness': self._assess_transformation_readiness(state),
            'recommendations': recommendations
        }
    
    def _classify_deviation_pattern(self, state, manifold_result, attractor_result) -> str:
        """乖離パターンの分類"""
        
        # Shadow活性化パターン
        if state.Λ_self_aspects['shadow'] > 0.6:
            if state.metacognition > 0.5:
                return "conscious_shadow_integration"
            else:
                return "unconscious_shadow_eruption"
        
        # 成長的緊張パターン
        if state.ρT > 0.7 and state.Λ_self_aspects['ideal'] > state.Λ_self_aspects['core']:
            return "growth_tension"
        
        # 創造的不均衡パターン
        if manifold_result['topological_features']['connectivity'] > 0.7:
            return "creative_imbalance"
        
        # 固着ループパターン
        if attractor_result['nearest_attractor'] and \
           attractor_result['nearest_attractor']['attractor'].attractor_type == 'pathological':
            return "fixation_loop"
        
        # 機能崩壊パターン
        if state.Λ_self_aspects['core'] < 0.4:
            return "functional_collapse"
        
        return "transitional_state"
    
    def _evaluate_growth_potential(self, state, pattern: str, attractor_result) -> float:
        """成長可能性の評価"""
        
        potential = 0.5  # 基準値
        
        # パターン別の基本ポテンシャル
        pattern_potentials = {
            "conscious_shadow_integration": 0.9,
            "growth_tension": 0.8,
            "creative_imbalance": 0.7,
            "transitional_state": 0.5,
            "unconscious_shadow_eruption": 0.3,
            "fixation_loop": 0.2,
            "functional_collapse": 0.1
        }
        
        potential = pattern_potentials.get(pattern, 0.5)
        
        # 修正要因
        # メタ認知が高ければポテンシャル上昇
        potential += state.metacognition * 0.2
        
        # 健全なアトラクターが近ければ上昇
        if attractor_result['trajectory_prediction'].get('target_type') == 'healthy':
            potential += 0.1
        
        # リソース（Λ_bod）があれば上昇
        potential += state.Λ_bod * 0.1
        
        return np.clip(potential, 0.0, 1.0)
    
    def _determine_deviation_quality(self, pattern: str, 
                                   growth_potential: float,
                                   state) -> str:
        """建設的/破壊的の判定"""
        
        # 建設的パターン
        constructive_patterns = [
            "conscious_shadow_integration",
            "growth_tension",
            "creative_imbalance"
        ]
        
        # 破壊的パターン
        destructive_patterns = [
            "fixation_loop",
            "functional_collapse"
        ]
        
        if pattern in constructive_patterns:
            return "constructive"
        elif pattern in destructive_patterns:
            return "destructive"
        else:
            # 成長可能性で判定
            if growth_potential > 0.6:
                return "potentially_constructive"
            elif growth_potential < 0.3:
                return "potentially_destructive"
            else:
                return "ambiguous"
    
    def _identify_constructive_aspects(self, state) -> List[str]:
        """建設的側面の特定"""
        aspects = []
        
        # 高いメタ認知
        if state.metacognition > 0.6:
            aspects.append("high_self_awareness")
        
        # Shadow統合の兆し
        if 0.4 < state.Λ_self_aspects['shadow'] < 0.6:
            aspects.append("shadow_integration_process")
        
        # 適応的柔軟性
        if state.perception_noise > 0.3 and state.metacognition > 0.5:
            aspects.append("adaptive_flexibility")
        
        # 創造的緊張
        if state.ρT > 0.6 and state.emotional_state.joy > 0.5:
            aspects.append("creative_tension")
        
        return aspects
    
    def _identify_destructive_aspects(self, state) -> List[str]:
        """破壊的側面の特定"""
        aspects = []
        
        # Core自己の弱体化
        if state.Λ_self_aspects['core'] < 0.5:
            aspects.append("weakened_core_self")
        
        # Shadow支配
        if state.Λ_self_aspects['shadow'] > state.Λ_self_aspects['core']:
            aspects.append("shadow_dominance")
        
        # メタ認知の崩壊
        if state.metacognition < 0.3:
            aspects.append("metacognitive_collapse")
        
        # 社会的孤立
        if state.σs < 0.3:
            aspects.append("social_isolation")
        
        # 身体的枯渇
        if state.Λ_bod < 0.4:
            aspects.append("physical_depletion")
        
        return aspects
    
    def _assess_transformation_readiness(self, state) -> float:
        """変容への準備度評価"""
        readiness = 0.0
        
        # 危機認識
        if state.developmental_stage == DevelopmentalStage.CRISIS:
            readiness += 0.2
        
        # 十分なリソース
        if state.Λ_bod > 0.5 and state.Λ_self > 0.5:
            readiness += 0.3
        
        # メタ認知能力
        readiness += state.metacognition * 0.3
        
        # 柔軟性（適度なノイズ）
        if 0.2 < state.perception_noise < 0.5:
            readiness += 0.2
        
        return readiness
    
    def _generate_quality_based_recommendations(self, quality_type: str,
                                              pattern: str) -> List[str]:
        """質に基づく推奨事項"""
        recommendations = []
        
        if quality_type == "constructive":
            recommendations.extend([
                "現在の成長プロセスを信頼し、支援的に見守る",
                "自己探求のためのスペースと時間を確保",
                "創造的な表現活動を促進"
            ])
        elif quality_type == "potentially_constructive":
            recommendations.extend([
                "メタ認知能力を高める練習（日記、瞑想）",
                "安全な環境での実験的行動を奨励",
                "定期的な振り返りとフィードバック"
            ])
        elif quality_type == "destructive":
            recommendations.extend([
                "専門的サポートの早急な導入",
                "安定化を最優先（睡眠、食事、ルーティン）",
                "小さな成功体験の積み重ね"
            ])
        elif quality_type == "potentially_destructive":
            recommendations.extend([
                "予防的介入の検討",
                "サポートネットワークの強化",
                "ストレス要因の特定と軽減"
            ])
        else:  # ambiguous
            recommendations.extend([
                "継続的なモニタリング",
                "多角的なアセスメントの実施",
                "柔軟な対応準備"
            ])
        
        # パターン特有の推奨
        pattern_specific = {
            "conscious_shadow_integration": "統合プロセスを深めるための深層心理ワーク",
            "growth_tension": "成長の方向性を明確にするコーチング",
            "fixation_loop": "パターン中断のための行動療法的介入",
            "functional_collapse": "機能回復のための段階的リハビリテーション"
        }
        
        if pattern in pattern_specific:
            recommendations.append(pattern_specific[pattern])
        
        return recommendations

# =========== 新規追加：拡張機能 ===========

@dataclass
class PathInteraction:
    """❶ PATH相互作用テンソル"""
    interaction_matrix: np.ndarray = field(default_factory=lambda: np.eye(8))
    resonance_factors: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    def __post_init__(self):
        # INTJ特有の相互作用パターンを初期化
        self._initialize_intj_interactions()
    
    def _initialize_intj_interactions(self):
        """INTJの機能間相互作用を設定"""
        # Ni-Te: 強い正の相関（主機能-補助機能）
        self.set_interaction('Ni', 'Te', 0.8)
        # Ni-Se: 負の相関（主機能-劣等機能）
        self.set_interaction('Ni', 'Se', -0.6)
        # Fi-Te: 緊張関係
        self.set_interaction('Fi', 'Te', -0.3)
        # Shadow functions interactions
        self.set_interaction('Ne', 'Ni', -0.7)  # Shadow対立
        self.set_interaction('Ti', 'Te', -0.5)
        self.set_interaction('Fe', 'Fi', -0.4)
        self.set_interaction('Si', 'Se', -0.6)
    
    def set_interaction(self, path1: str, path2: str, strength: float):
        """PATH間の相互作用を設定"""
        idx1 = BASE_PATHS.index(path1)
        idx2 = BASE_PATHS.index(path2)
        self.interaction_matrix[idx1, idx2] = strength
        self.interaction_matrix[idx2, idx1] = strength
        self.resonance_factors[(path1, path2)] = strength
    
    def calculate_path_influence(self, active_path: str, path_states: Dict[str, float]) -> float:
        """特定PATHが他のPATHから受ける影響を計算"""
        idx = BASE_PATHS.index(active_path)
        influence = 0.0
        
        for i, other_path in enumerate(BASE_PATHS):
            if other_path != active_path:
                interaction_strength = self.interaction_matrix[idx, i]
                influence += interaction_strength * path_states.get(other_path, 0)
        
        return influence

@dataclass
class PulsationEvent:
    """❷ 拍動イベント"""
    timestamp: int
    intensity: float
    trigger_path: str
    emotional_result: str
    tensor_snapshot: Dict[str, float]

class DynamicLambda3State:
    """拡張版Λ³状態（動的相互作用対応）"""
    
    def __init__(self):
        # 基本テンソル
        self.Λ_self = 0.8
        self.σs = 0.5
        self.ρT = 0.6
        self.ΛF = np.array([0.7, 0.3, 0.5])
        self.Λ_bod = 0.7
        self.ε_critical = 0.5  # 臨界閾値
        
        # 感情と欲求
        self.emotional_state = EmotionalState()
        self.current_desire = Desire()
        
        # 自己定義の多面性
        self.Λ_self_aspects = {
            'core': 0.8,
            'ideal': 0.7,
            'social': 0.6,
            'shadow': 0.3
        }
        
        # PATH関連
        self.path_preferences = {
            path: PathPreference.NEUTRAL for path in BASE_PATHS
        }
        self.path_states = {path: 0.5 for path in BASE_PATHS}
        self.path_trauma_count = {path: 0 for path in BASE_PATHS}
        
        # メタ状態
        self.developmental_stage = DevelopmentalStage.FORMATION
        self.stage_duration = 0
        self.perception_noise = 0.1
        self.metacognition = 0.5
        
        # ❶ PATH相互作用テンソル
        self.path_interaction = PathInteraction()
        
        # ❷ Pulsationイベント履歴
        self.pulsation_history: List[PulsationEvent] = []
        self.time_step = 0
        
        # ❸ 認識ノイズとメタ認知の相互作用パラメータ
        self.noise_metacognition_coupling = 0.05
        
        # ❹ 社会的ネットワーク
        self.social_network: Dict[str, List[SocialFeedback]] = {}
        
        # ❺ イベントキュー
        self.event_queue: List[Event] = []
        self.active_events: List[Tuple[Event, int]] = []  # (event, remaining_duration)
    
    def check_pulsation_conditions(self) -> bool:
        """❷ 拍動条件のチェック"""
        return (self.Λ_self > 0 and 
                self.σs < 1.0 and 
                self.ρT > self.ε_critical and 
                np.linalg.norm(self.ΛF) > 0)
    
    def trigger_pulsation(self, trigger_path: Optional[str] = None) -> Optional[PulsationEvent]:
        """❷ 拍動イベントの生成"""
        if not self.check_pulsation_conditions():
            return None
        
        # 拍動強度の計算
        intensity = self.ρT * self.σs * np.linalg.norm(self.ΛF)
        
        # 感情結果の決定（簡略化）
        dominant = self.emotional_state.dominant_emotion()
        emotional_result = dominant[0]
        
        # スナップショット作成
        snapshot = {
            'Λ_self': self.Λ_self,
            'σs': self.σs,
            'ρT': self.ρT,
            'perception_noise': self.perception_noise,
            'metacognition': self.metacognition
        }
        
        event = PulsationEvent(
            timestamp=self.time_step,
            intensity=intensity,
            trigger_path=trigger_path or "general",
            emotional_result=emotional_result,
            tensor_snapshot=snapshot
        )
        
        self.pulsation_history.append(event)
        
        # 拍動による状態更新
        self._update_state_after_pulsation(event)
        
        return event
    
    def _update_state_after_pulsation(self, event: PulsationEvent):
        """拍動後の状態更新"""
        # テンション解放
        self.ρT *= 0.7
        
        # メタ認知への影響
        if event.intensity > 1.0:
            self.metacognition = min(1.0, self.metacognition + 0.05)
        
        # Shadow側面への影響
        if event.emotional_result in ['anger', 'fear']:
            self.Λ_self_aspects['shadow'] = min(0.8, 
                self.Λ_self_aspects['shadow'] + 0.05)
    
    def update_metacognition_and_noise(self):
        """❸ 認識ノイズとメタ認知の動的相互作用"""
        # ノイズがメタ認知を妨げる
        self.metacognition -= self.perception_noise * self.noise_metacognition_coupling
        self.metacognition = np.clip(self.metacognition, 0.1, 1.0)
        
        # メタ認知がノイズを抑制
        self.perception_noise -= self.metacognition * 0.03
        self.perception_noise = np.clip(self.perception_noise, 0.0, 1.0)
        
        # PATH相互作用による追加ノイズ
        path_conflict = self._calculate_path_conflict()
        self.perception_noise += path_conflict * 0.02
    
    def _calculate_path_conflict(self) -> float:
        """PATH間の葛藤度を計算"""
        conflict = 0.0
        for path1 in BASE_PATHS[:4]:  # 主要機能
            for path2 in BASE_PATHS[4:]:  # Shadow機能
                if self.path_states[path1] > 0.6 and self.path_states[path2] > 0.6:
                    interaction = self.path_interaction.interaction_matrix[
                        BASE_PATHS.index(path1), BASE_PATHS.index(path2)
                    ]
                    if interaction < 0:
                        conflict += abs(interaction) * self.path_states[path1] * self.path_states[path2]
        return conflict
    
    def add_social_feedback(self, feedback: SocialFeedback):
        """❹ 社会的フィードバックの追加（ネットワークモデル）"""
        if feedback.source_id not in self.social_network:
            self.social_network[feedback.source_id] = []
        
        self.social_network[feedback.source_id].append(feedback)
        
        # ネットワーク効果の計算
        self._process_network_feedback(feedback)
    
    def _process_network_feedback(self, new_feedback: SocialFeedback):
        """ネットワーク全体からの影響を計算"""
        total_impact = 0.0
        authenticity_weighted_sum = 0.0
        
        for source_id, feedbacks in self.social_network.items():
            for fb in feedbacks[-5:]:  # 最新5つのフィードバック
                weight = fb.authenticity * fb.intensity
                total_impact += fb.valence * weight
                authenticity_weighted_sum += weight
        
        if authenticity_weighted_sum > 0:
            network_effect = total_impact / authenticity_weighted_sum
            
            # 社会的自己の更新
            self.Λ_self_aspects['social'] += network_effect * 0.05
            self.Λ_self_aspects['social'] = np.clip(self.Λ_self_aspects['social'], 0.1, 1.0)
            
            # 同期率への影響
            if network_effect > 0.5:
                self.σs = min(0.9, self.σs + 0.02)
            elif network_effect < -0.5:
                self.σs = max(0.1, self.σs - 0.02)
    
    def process_event(self, event: Event):
        """❺ イベントの動的統合"""
        # 即時影響
        immediate_impact = event.impact * (1 - event.uncertainty)
        delayed_impact = event.impact * event.uncertainty
        
        # Core自己への即時影響
        self.Λ_self_aspects['core'] += immediate_impact * 0.1
        self.Λ_self_aspects['core'] = np.clip(self.Λ_self_aspects['core'], 0.1, 1.0)
        
        # テンション密度への影響
        self.ρT += abs(delayed_impact) * 0.3
        self.ρT = min(1.0, self.ρT)
        
        # 持続的影響のキューイング
        if event.duration > 0:
            self.active_events.append((event, event.duration))
        
        # イベントタイプによる特異的影響
        self._apply_event_specific_effects(event)
    
    def _apply_event_specific_effects(self, event: Event):
        """イベントタイプ特有の影響を適用"""
        if event.event_type == 'loss':
            self.emotional_state.sadness += 0.2
            self.current_desire.security += 0.1
        elif event.event_type == 'opportunity':
            self.emotional_state.joy += 0.1
            self.current_desire.achievement += 0.2
        elif event.event_type == 'conflict':
            self.emotional_state.anger += 0.15
            self.σs -= 0.05
    
    def update_path_states_with_interaction(self):
        """PATH状態を相互作用を考慮して更新"""
        new_states = {}
        
        for path in BASE_PATHS:
            # 基本状態
            base_state = self.path_states[path]
            
            # 他のPATHからの影響
            influence = self.path_interaction.calculate_path_influence(
                path, self.path_states
            )
            
            # 嗜好による調整
            preference_modifier = {
                PathPreference.AVOIDED: -0.3,
                PathPreference.RELUCTANT: -0.1,
                PathPreference.NEUTRAL: 0.0,
                PathPreference.PREFERRED: 0.2,
                PathPreference.DEPENDENT: 0.4
            }
            pref_mod = preference_modifier[self.path_preferences[path]]
            
            # 新しい状態の計算
            new_state = base_state + influence * 0.1 + pref_mod * 0.05
            new_states[path] = np.clip(new_state, 0.0, 1.0)
        
        self.path_states = new_states
    
    def time_step_update(self):
        """時間ステップごとの更新"""
        self.time_step += 1
        
        # PATH相互作用の更新
        self.update_path_states_with_interaction()
        
        # ノイズとメタ認知の相互作用
        self.update_metacognition_and_noise()
        
        # 持続的イベントの処理
        remaining_events = []
        for event, duration in self.active_events:
            if duration > 1:
                remaining_events.append((event, duration - 1))
                # 持続的影響
                self.ρT += event.impact * 0.05
                
        self.active_events = remaining_events
        
        # 拍動チェック
        if self.check_pulsation_conditions() and random.random() > 0.7:
            self.trigger_pulsation()

# =========== 拡張版評価システム ===========

class EnhancedMBTIAssessment:
    """拡張版MBTI評価システム（動的相互作用対応）"""
    
    def __init__(self):
        self.mbti_stacks = {
            'INTJ': MBTIProfile('INTJ', 'Ni', 'Te', 'Fi', 'Se')
        }
    
    def generate_enhanced_questions(self) -> Dict:
        """拡張版質問セット（イベントベース質問を含む）"""
        
        # 既存の質問（完全版）
        basic_questions = {
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
        
        # 新規：イベントベース質問
        event_questions = {
            'pulsation_events': [
                {
                    'id': 'pulse_1',
                    'text': '感情が急激に高まって、涙が出たり声が震えたりすることがありますか？',
                    'scale': '1:全くない 2:月1回 3:週1回 4:週2-3回 5:ほぼ毎日',
                    'weight': 1.2,
                    'type': 'pulsation_frequency'
                },
                {
                    'id': 'pulse_2',
                    'text': 'その感情の高まりの後、しばらく疲労感や虚無感を感じますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'type': 'post_pulsation'
                }
            ],
            'path_conflicts': [
                {
                    'id': 'conflict_1',
                    'text': '論理的に考えたいのに感情に振り回される（またはその逆）ことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.9,
                    'type': 'te_fi_conflict'
                },
                {
                    'id': 'conflict_2',
                    'text': '内なるビジョンと現実の要求の間で引き裂かれる感覚がありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 1.0,
                    'type': 'ni_se_conflict'
                }
            ],
            'social_network_effects': [
                {
                    'id': 'social_1',
                    'text': '他者からのフィードバックによって自己評価が大きく揺らぐことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.7,
                    'type': 'social_impact'
                },
                {
                    'id': 'social_2',
                    'text': '複数の人から矛盾するフィードバックを受けて混乱することがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'type': 'network_confusion'
                }
            ],
            'metacognition_noise': [
                {
                    'id': 'meta_1',
                    'text': '自分の思考や感情を客観的に観察できなくなることがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.9,
                    'type': 'metacognition_loss'
                },
                {
                    'id': 'meta_2',
                    'text': '頭の中がノイズや雑念でいっぱいになり、集中できないことがありますか？',
                    'scale': '1:全くない 2:たまに 3:時々 4:頻繁に 5:常に',
                    'weight': 0.8,
                    'type': 'perception_noise'
                }
            ],
            'critical_events': [
                {
                    'id': 'event_1',
                    'text': '最近、大きな喪失体験（離別、失敗、挫折など）がありましたか？',
                    'scale': '1:なし 2:小さい 3:中程度 4:大きい 5:非常に大きい',
                    'weight': 1.5,
                    'type': 'loss_event'
                },
                {
                    'id': 'event_2',
                    'text': 'その出来事の影響は今も続いていますか？',
                    'scale': '1:全くない 2:少し 3:中程度 4:かなり 5:非常に強く',
                    'weight': 1.2,
                    'type': 'event_duration'
                }
            ]
        }
        
        # 統合
        all_questions = {**basic_questions, **event_questions}
        return all_questions
    
    def calculate_dysfunction_scores(self, responses: Dict[str, int]) -> Dict:
        """回答から機能不全スコアを計算（元の実装を統合）"""
        
        questions = self.generate_enhanced_questions()
        
        # PATHごとのスコア集計
        path_scores = {
            'Ni': {'overuse': 0, 'isolation': 0, 'total_weight': 0},
            'Te': {'dysfunction': 0, 'suppression': 0, 'total_weight': 0},
            'Fi': {'eruption': 0, 'rigidity': 0, 'total_weight': 0},
            'Se': {'inferior_grip': 0, 'somatic': 0, 'total_weight': 0},
            'Ne': {'shadow': 0, 'total_weight': 0},
            'Ti': {'shadow': 0, 'total_weight': 0},
            'Fe': {'shadow': 0, 'total_weight': 0},
            'Si': {'shadow': 0, 'total_weight': 0},
            'general': {'imbalance': 0, 'total_weight': 0},
            'physical': {'dysfunction': 0, 'total_weight': 0}
        }
        
        # 各質問カテゴリのスコア計算
        for category, question_list in questions.items():
            # イベントベース質問はスキップ（別途処理）
            if category in ['pulsation_events', 'path_conflicts', 'social_network_effects', 
                           'metacognition_noise', 'critical_events']:
                continue
                
            for q in question_list:
                if q['id'] in responses:
                    score = responses[q['id']]
                    weighted_score = (score - 1) / 4 * q['weight']  # 0-1に正規化
                    
                    path = q['path']
                    pattern = q['pattern']
                    
                    if path in path_scores:
                        if pattern in ['overuse', 'dysfunction', 'eruption', 'inferior_grip', 
                                      'isolation', 'suppression', 'rigidity', 'somatic']:
                            path_scores[path][pattern] = path_scores[path].get(pattern, 0) + weighted_score
                        elif pattern.startswith('shadow'):
                            path_scores[path]['shadow'] = path_scores[path].get('shadow', 0) + weighted_score
                        elif pattern in ['sleep_deprivation', 'sleep_quality', 'meal_irregularity', 'eating_disorder']:
                            path_scores[path]['dysfunction'] = path_scores[path].get('dysfunction', 0) + weighted_score
                        elif pattern in ['dissociation', 'regression']:
                            if path == 'general':
                                path_scores[path]['imbalance'] = path_scores[path].get('imbalance', 0) + weighted_score
                        
                        path_scores[path]['total_weight'] += q['weight']
        
        # 正規化
        normalized_scores = {}
        for path, scores in path_scores.items():
            if scores['total_weight'] > 0:
                for pattern, value in scores.items():
                    if pattern != 'total_weight':
                        normalized_scores[f'{path}_{pattern}'] = value / scores['total_weight']
        
        return normalized_scores
    
    def apply_dysfunction_effects(self, state: DynamicLambda3State,
                                dysfunction_scores: Dict[str, float]) -> None:
        """機能不全スコアに基づく状態への影響適用"""
        
        # Ni過剰使用の影響
        if 'Ni_overuse' in dysfunction_scores:
            ni_overuse = dysfunction_scores['Ni_overuse']
            if ni_overuse > 0.7:
                state.path_preferences['Ni'] = PathPreference.DEPENDENT
                state.perception_noise += ni_overuse * 0.3
                state.Λ_self_aspects['ideal'] = min(0.95, state.Λ_self_aspects['ideal'] + ni_overuse * 0.1)
        
        # Te機能不全の影響
        if 'Te_dysfunction' in dysfunction_scores:
            te_dysfunction = dysfunction_scores['Te_dysfunction']
            if te_dysfunction > 0.6:
                state.path_preferences['Te'] = PathPreference.RELUCTANT
                state.Λ_self_aspects['core'] *= (1 - te_dysfunction * 0.2)
                state.Λ_self_aspects['social'] *= (1 - te_dysfunction * 0.1)
        
        # Fi爆発の影響
        if 'Fi_eruption' in dysfunction_scores:
            fi_eruption = dysfunction_scores['Fi_eruption']
            if fi_eruption > 0.5:
                state.path_preferences['Fi'] = PathPreference.PREFERRED
                state.emotional_state.anger += fi_eruption * 0.3
                state.emotional_state.sadness += fi_eruption * 0.2
        
        # Se劣等機能の暴走
        if 'Se_inferior_grip' in dysfunction_scores:
            se_grip = dysfunction_scores['Se_inferior_grip']
            if se_grip > 0.5:
                state.path_preferences['Se'] = PathPreference.AVOIDED
                state.Λ_self_aspects['shadow'] = min(0.8, 
                    state.Λ_self_aspects['shadow'] + se_grip * 0.3)
                state.σs *= (1 - se_grip * 0.3)
                state.ρT = min(0.95, state.ρT + se_grip * 0.4)
                state.Λ_bod *= (1 - se_grip * 0.2)
        
        # 全体的な不均衡
        if 'general_imbalance' in dysfunction_scores:
            imbalance = dysfunction_scores['general_imbalance']
            if imbalance > 0.6:
                state.metacognition *= (1 - imbalance * 0.3)
                for path in ['Ni', 'Te', 'Fi', 'Se']:
                    state.path_trauma_count[path] += int(imbalance * 5)
        
        # シャドウ機能の影響
        shadow_scores = {
            'Ne': dysfunction_scores.get('Ne_shadow', 0),
            'Ti': dysfunction_scores.get('Ti_shadow', 0),
            'Fe': dysfunction_scores.get('Fe_shadow', 0),
            'Si': dysfunction_scores.get('Si_shadow', 0)
        }
        
        total_shadow_activation = sum(shadow_scores.values()) / 4
        
        if total_shadow_activation > 0.5:
            state.Λ_self_aspects['shadow'] = min(0.9, 
                state.Λ_self_aspects['shadow'] + total_shadow_activation * 0.4)
            state.perception_noise += total_shadow_activation * 0.2
            
            # 各シャドウ機能の特異的影響
            if shadow_scores['Ne'] > 0.6:
                state.path_preferences['Ne'] = PathPreference.DEPENDENT
                state.ρT = min(0.95, state.ρT + 0.3)
                
            if shadow_scores['Ti'] > 0.6:
                state.path_preferences['Ti'] = PathPreference.PREFERRED
                state.σs *= 0.7
                
            if shadow_scores['Fe'] > 0.6:
                state.path_preferences['Fe'] = PathPreference.DEPENDENT
                state.Λ_self_aspects['social'] *= 0.8
                
            if shadow_scores['Si'] > 0.6:
                state.path_preferences['Si'] = PathPreference.PREFERRED
                state.Λ_bod *= 0.8
        
        # 身体的要因の処理
        if 'physical_dysfunction' in dysfunction_scores:
            physical_dysfunction = dysfunction_scores['physical_dysfunction']
            if physical_dysfunction > 0.5:
                state.Λ_bod *= (1 - physical_dysfunction * 0.3)
                state.perception_noise += physical_dysfunction * 0.1
                state.metacognition *= (1 - physical_dysfunction * 0.1)
                
                # 重度の身体的不調は発達段階にも影響
                if physical_dysfunction > 0.7:
                    state.developmental_stage = DevelopmentalStage.CRISIS
    
    def apply_enhanced_dynamics(self, state: DynamicLambda3State, 
                               responses: Dict[str, int]) -> None:
        """拡張された動的効果の適用"""
        
        # Pulsationイベントの生成
        if responses.get('pulse_1', 0) >= 4:
            # 高頻度の拍動
            for _ in range(3):
                state.trigger_pulsation('emotional_overflow')
                
        # PATH葛藤の処理
        if responses.get('conflict_1', 0) >= 4:
            # Te-Fi葛藤
            state.path_interaction.set_interaction('Te', 'Fi', -0.6)
            state.perception_noise += 0.1
            
        if responses.get('conflict_2', 0) >= 4:
            # Ni-Se葛藤
            state.path_interaction.set_interaction('Ni', 'Se', -0.8)
            state.ρT += 0.2
        
        # 社会的ネットワーク効果
        if responses.get('social_1', 0) >= 3:
            # 不安定な社会的自己
            feedback = SocialFeedback(
                source_id='assessment',
                valence=-0.3,
                intensity=0.7,
                authenticity=0.8,
                category='criticism'
            )
            state.add_social_feedback(feedback)
        
        # メタ認知とノイズ
        if responses.get('meta_1', 0) >= 4:
            state.metacognition *= 0.7
        if responses.get('meta_2', 0) >= 4:
            state.perception_noise += 0.2
        
        # 重大イベントの処理
        if responses.get('event_1', 0) >= 4:
            event = Event(
                event_type='loss',
                impact=-0.8,
                uncertainty=0.3,
                duration=responses.get('event_2', 3) * 10
            )
            state.process_event(event)
    
    def calculate_enhanced_evaluation(self, state: DynamicLambda3State, 
                                    responses: Dict[str, int]) -> Dict:
        """拡張版評価の計算"""
        
        # 元の機能不全スコアを計算
        dysfunction_scores = self.calculate_dysfunction_scores(responses)
        
        # 基本評価指標（元の実装を考慮）
        indicators = {
            'core_stability': state.Λ_self_aspects['core'] > 0.5,
            'shadow_integration': state.Λ_self_aspects['shadow'] < state.Λ_self_aspects['core'],
            'social_coherence': state.Λ_self_aspects['social'] > 0.4,
            'metacognitive_function': state.metacognition > 0.4,
            'noise_control': state.perception_noise < 0.3,
            'tension_regulation': state.ρT < 0.8,
            'bodily_vitality': state.Λ_bod > 0.5,
            'pulsation_balance': len(state.pulsation_history) < 10,
            'ni_overuse_control': dysfunction_scores.get('Ni_overuse', 0) < 0.7,
            'se_grip_control': dysfunction_scores.get('Se_inferior_grip', 0) < 0.6,
            'shadow_activation_control': sum([
                dysfunction_scores.get('Ne_shadow', 0),
                dysfunction_scores.get('Ti_shadow', 0),
                dysfunction_scores.get('Fe_shadow', 0),
                dysfunction_scores.get('Si_shadow', 0)
            ]) / 4 < 0.5
        }
        
        # PATH相互作用の健全性
        path_health = self._evaluate_path_interactions(state)
        
        # ネットワーク効果
        network_health = self._evaluate_social_network(state)
        
        # 総合スコア
        basic_score = sum(1 for v in indicators.values() if v)
        interaction_score = path_health['score']
        network_score = network_health['score']
        
        total_score = basic_score + interaction_score + network_score
        
        # 重症度判定（より細かい基準）
        if total_score <= 4:
            severity = "severe"
            interpretation = "重度の機能不全・緊急介入必要"
        elif total_score <= 7:
            severity = "moderate-severe"
            interpretation = "中等度から重度の機能不全"
        elif total_score <= 10:
            severity = "moderate"
            interpretation = "中等度の機能不全"
        elif total_score <= 12:
            severity = "mild"
            interpretation = "軽度の機能不全"
        else:
            severity = "healthy"
            interpretation = "健康的な状態"
        
        # 主要な問題の特定
        primary_issues = []
        if dysfunction_scores.get('Ni_overuse', 0) > 0.6:
            primary_issues.append("Ni過剰使用による現実逃避")
        if dysfunction_scores.get('Se_inferior_grip', 0) > 0.5:
            primary_issues.append("Se劣等機能の暴走")
        if not indicators['metacognitive_function']:
            primary_issues.append("メタ認知の著しい低下")
        if dysfunction_scores.get('general_imbalance', 0) > 0.7:
            primary_issues.append("全体的な心理的不均衡")
        if not indicators['shadow_activation_control']:
            primary_issues.append("シャドウ機能の活性化による人格の分裂傾向")
        
        # ========== 高度分析の実行 ==========
        # マニフォールド解析
        manifold_analysis = self.perform_manifold_analysis(state, dysfunction_scores)
        
        # アトラクター解析
        attractor_analysis = self.perform_attractor_analysis(state, 'INTJ')
        
        # 乖離の質的分析
        deviation_quality = self.perform_deviation_quality_analysis(
            state, manifold_analysis, attractor_analysis
        )
        
        # 高度分析結果を統合
        advanced_analysis = {
            'manifold': manifold_analysis,
            'attractor': attractor_analysis,
            'deviation_quality': deviation_quality
        }
        
        return {
            'severity': severity,
            'interpretation': interpretation,
            'total_score': total_score,
            'max_score': 16,  # 基本11 + PATH相互作用3 + ネットワーク2
            'indicators': indicators,
            'dysfunction_scores': dysfunction_scores,
            'path_health': path_health,
            'network_health': network_health,
            'pulsation_analysis': self._analyze_pulsations(state),
            'primary_issues': primary_issues,
            'recommendations': self._generate_recommendations(state, severity, dysfunction_scores),
            'advanced_analysis': advanced_analysis  # 新規追加
        }
    
    def perform_manifold_analysis(self, state: DynamicLambda3State,
                                dysfunction_scores: Dict[str, float]) -> Dict:
        """Λ³マニフォールド解析の実行"""
        analyzer = Lambda3ManifoldAnalyzer()
        return analyzer.analyze(state, dysfunction_scores)
    
    def perform_attractor_analysis(self, state: DynamicLambda3State,
                                 mbti_type: str) -> Dict:
        """アトラクター盆地解析の実行"""
        analyzer = AttractorBasinAnalyzer(mbti_type)
        return analyzer.analyze(state)
    
    def perform_deviation_quality_analysis(self, state: DynamicLambda3State,
                                         manifold_result: Dict,
                                         attractor_result: Dict) -> Dict:
        """乖離の質的分析の実行"""
        analyzer = DeviationQualityAnalyzer()
        return analyzer.analyze(state, manifold_result, attractor_result)
    
    def _evaluate_path_interactions(self, state: DynamicLambda3State) -> Dict:
        """PATH相互作用の健全性評価"""
        conflicts = 0
        synergies = 0
        
        for (path1, path2), strength in state.path_interaction.resonance_factors.items():
            if strength < -0.5:
                conflicts += 1
            elif strength > 0.5:
                synergies += 1
        
        score = max(0, 3 - conflicts + synergies // 2)
        
        return {
            'score': min(score, 3),
            'conflicts': conflicts,
            'synergies': synergies
        }
    
    def _evaluate_social_network(self, state: DynamicLambda3State) -> Dict:
        """社会的ネットワークの健全性評価"""
        if not state.social_network:
            return {'score': 1, 'sources': 0, 'balance': 0}
        
        positive_count = 0
        negative_count = 0
        
        for feedbacks in state.social_network.values():
            for fb in feedbacks:
                if fb.valence > 0:
                    positive_count += 1
                else:
                    negative_count += 1
        
        balance = positive_count / (positive_count + negative_count + 1)
        score = 2 if 0.3 < balance < 0.7 else 1
        
        return {
            'score': score,
            'sources': len(state.social_network),
            'balance': balance
        }
    
    def _analyze_pulsations(self, state: DynamicLambda3State) -> Dict:
        """拍動パターンの分析"""
        if not state.pulsation_history:
            return {'frequency': 0, 'average_intensity': 0, 'pattern': 'none'}
        
        recent_pulsations = state.pulsation_history[-10:]
        frequency = len(recent_pulsations)
        avg_intensity = np.mean([p.intensity for p in recent_pulsations])
        
        if frequency > 7:
            pattern = 'hyperactive'
        elif frequency > 4:
            pattern = 'active'
        elif frequency > 1:
            pattern = 'moderate'
        else:
            pattern = 'suppressed'
        
        return {
            'frequency': frequency,
            'average_intensity': avg_intensity,
            'pattern': pattern
        }
    
    def _generate_recommendations(self, state: DynamicLambda3State, 
                                severity: str, 
                                dysfunction_scores: Dict[str, float]) -> List[str]:
        """動的状態と機能不全スコアに基づく推奨事項"""
        recommendations = []
        
        # 基本推奨
        if severity in ["severe", "moderate-severe"]:
            recommendations.append("専門家（精神科医・心理士）への相談を強く推奨")
        
        # Ni過剰使用への対処
        if dysfunction_scores.get('Ni_overuse', 0) > 0.6:
            recommendations.append("具体的な行動計画の作成と実行（小さなステップから）")
            recommendations.append("現実的な目標設定の練習")
        
        # Se劣等機能暴走への対処
        if dysfunction_scores.get('Se_inferior_grip', 0) > 0.5:
            recommendations.append("健全な感覚体験の構造化（運動、自然散策など）")
            recommendations.append("衝動的行動の前に一時停止する練習")
        
        # PATH相互作用に基づく推奨
        if state.path_interaction.resonance_factors.get(('Ni', 'Se'), 0) < -0.7:
            recommendations.append("Ni-Se統合のための身体的グラウンディング練習")
            recommendations.append("マインドフルネスや瞑想での現在への注目")
        
        # メタ認知に基づく推奨
        if state.metacognition < 0.4:
            recommendations.append("日記や内省による自己観察力の回復")
            recommendations.append("認知行動療法的アプローチの検討")
        
        # シャドウ機能への対処
        shadow_activation = sum([
            dysfunction_scores.get('Ne_shadow', 0),
            dysfunction_scores.get('Ti_shadow', 0),
            dysfunction_scores.get('Fe_shadow', 0),
            dysfunction_scores.get('Si_shadow', 0)
        ]) / 4
        
        if shadow_activation > 0.5:
            recommendations.append("シャドウワーク：抑圧された側面との対話")
            recommendations.append("夢分析やアクティブイマジネーション")
            recommendations.append("統合的な心理療法（ユング派分析など）の検討")
        
        # 社会的ネットワークに基づく推奨
        if len(state.social_network) < 2:
            recommendations.append("信頼できる支援者との関係構築")
        
        # 拍動パターンに基づく推奨
        pulsation_analysis = self._analyze_pulsations(state)
        if pulsation_analysis['pattern'] == 'hyperactive':
            recommendations.append("感情調整スキルの学習（DBTなど）")
            recommendations.append("定期的な休息と回復時間の確保")
        elif pulsation_analysis['pattern'] == 'suppressed':
            recommendations.append("感情表現の安全な練習")
            recommendations.append("身体感覚への注目（ボディスキャン等）")
        
        # 身体的要因への対処
        if dysfunction_scores.get('physical_dysfunction', 0) > 0.5:
            recommendations.append("睡眠衛生の改善（7-8時間の確保）")
            recommendations.append("規則正しい食事リズムの確立")
            recommendations.append("血糖値の安定化（タンパク質を含む朝食）")
        
        # 身体状態の低下への対処
        if state.Λ_bod < 0.5:
            recommendations.append("軽い運動習慣の導入（ウォーキング、ヨガなど）")
            recommendations.append("自然との接触時間を増やす")
        
        return recommendations

def run_enhanced_assessment():
    """拡張版評価の実行例"""
    print("=== 拡張版INTJ評価システム（Λ³動的相互作用モデル） ===\n")
    
    # システム初期化
    assessor = EnhancedMBTIAssessment()
    state = DynamicLambda3State()
    
    # サンプル回答（拡張版）
    sample_responses = {
        # 基本質問（完全版）
        'ni_1': 4, 'ni_2': 5, 'ni_3': 4,
        'te_1': 4, 'te_2': 3,
        'fi_1': 2, 'fi_2': 3,
        'se_1': 5, 'se_2': 4, 'se_3': 3,
        'gen_1': 4, 'gen_2': 4,
        # シャドウ機能
        'sh_ne_1': 3, 'sh_ne_2': 4,
        'sh_ti_1': 4, 'sh_ti_2': 3,
        'sh_fe_1': 3, 'sh_fe_2': 4,
        'sh_si_1': 5, 'sh_si_2': 4,
        # 生活習慣
        'sleep_1': 4, 'sleep_2': 4,
        'meal_1': 3, 'meal_2': 4,
        # イベントベース質問
        'pulse_1': 4,      # 高頻度の拍動
        'pulse_2': 4,
        'conflict_1': 5,   # 強いTe-Fi葛藤
        'conflict_2': 4,   # Ni-Se葛藤
        'social_1': 4,     # 社会的フィードバックに敏感
        'social_2': 3,
        'meta_1': 4,       # メタ認知の低下
        'meta_2': 5,       # 高い認識ノイズ
        'event_1': 4,      # 大きな喪失体験
        'event_2': 4       # 持続的影響
    }
    
    # 初期状態の表示
    print("初期状態:")
    print(f"  Core: {state.Λ_self_aspects['core']:.2f}")
    print(f"  Shadow: {state.Λ_self_aspects['shadow']:.2f}")
    print(f"  Metacognition: {state.metacognition:.2f}")
    print(f"  Perception Noise: {state.perception_noise:.2f}")
    
    # 機能不全スコアの計算と適用
    dysfunction_scores = assessor.calculate_dysfunction_scores(sample_responses)
    assessor.apply_dysfunction_effects(state, dysfunction_scores)
    
    # 動的効果の適用
    assessor.apply_enhanced_dynamics(state, sample_responses)
    
    # 数回の時間ステップ実行
    print("\n時系列シミュレーション:")
    for i in range(5):
        state.time_step_update()
        print(f"  Step {i+1}: ρT={state.ρT:.2f}, "
              f"Noise={state.perception_noise:.2f}, "
              f"Pulsations={len(state.pulsation_history)}")
    
    # 評価の実行
    evaluation = assessor.calculate_enhanced_evaluation(state, sample_responses)
    
    print(f"\n=== 拡張評価結果 ===")
    print(f"重症度: {evaluation['severity']}")
    print(f"解釈: {evaluation['interpretation']}")
    print(f"総合スコア: {evaluation['total_score']}/{evaluation['max_score']}")
    
    print("\n基本指標:")
    for key, value in evaluation['indicators'].items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}")
    
    print("\n主要な機能不全:")
    for key, score in evaluation['dysfunction_scores'].items():
        if score > 0.5:
            print(f"  • {key}: {score:.2f}")
    
    print(f"\nPATH相互作用:")
    print(f"  葛藤数: {evaluation['path_health']['conflicts']}")
    print(f"  協調数: {evaluation['path_health']['synergies']}")
    print(f"  スコア: {evaluation['path_health']['score']}/3")
    
    print(f"\n拍動分析:")
    print(f"  頻度: {evaluation['pulsation_analysis']['frequency']}")
    print(f"  平均強度: {evaluation['pulsation_analysis']['average_intensity']:.2f}")
    print(f"  パターン: {evaluation['pulsation_analysis']['pattern']}")
    
    if evaluation['primary_issues']:
        print("\n主要な問題:")
        for issue in evaluation['primary_issues']:
            print(f"  • {issue}")
    
    print("\n推奨事項:")
    for rec in evaluation['recommendations']:
        print(f"  • {rec}")
    
    # ========== 高度分析結果の表示 ==========
    if 'advanced_analysis' in evaluation:
        print("\n\n=== 高度分析結果 ===")
        
        # マニフォールド解析
        manifold = evaluation['advanced_analysis']['manifold']
        print("\n【Λ³マニフォールド解析】")
        print(f"  健全状態への測地線距離: {manifold['geodesic_distance_to_health']:.2f}")
        print(f"  局所曲率: {manifold['current_position'].curvature:.2f}")
        print(f"  位相速度: {manifold['phase_space_velocity']:.2f}")
        print(f"  近傍アトラクター: {', '.join(manifold['current_position'].nearby_attractors)}")
        if manifold['critical_points']:
            print("  臨界点:")
            for cp in manifold['critical_points']:
                print(f"    - {cp['description']} (深刻度: {cp['severity']:.2f})")
        
        # アトラクター解析
        attractor = evaluation['advanced_analysis']['attractor']
        print("\n【アトラクター盆地解析】")
        if attractor['nearest_attractor']:
            nearest = attractor['nearest_attractor']
            print(f"  最近接アトラクター: {nearest['attractor'].name}")
            print(f"    タイプ: {nearest['attractor'].attractor_type}")
            print(f"    距離: {nearest['distance']:.2f}")
            print(f"    盆地内: {'Yes' if nearest['in_basin'] else 'No'}")
        
        trajectory = attractor['trajectory_prediction']
        print(f"  予測軌道: {trajectory['direction']} → {trajectory['target']}")
        if trajectory['target']:
            print(f"    到達予想時間: {trajectory['estimated_time']:.1f}")
            print(f"    確信度: {trajectory['confidence']:.2f}")
        
        print(f"  安定性: {attractor['stability_analysis']['type']}")
        print(f"  脱出難易度: {attractor['escape_difficulty']:.2f}")
        
        # 乖離の質的分析
        quality = evaluation['advanced_analysis']['deviation_quality']
        deviation = quality['deviation_quality']
        print("\n【乖離の質的分析】")
        print(f"  乖離タイプ: {deviation.deviation_type}")
        print(f"  パターン: {deviation.pattern}")
        print(f"  成長可能性: {deviation.growth_potential:.2f}")
        
        if quality['constructive_aspects']:
            print(f"  建設的側面:")
            for aspect in quality['constructive_aspects']:
                print(f"    + {aspect}")
        
        if quality['destructive_aspects']:
            print(f"  破壊的側面:")
            for aspect in quality['destructive_aspects']:
                print(f"    - {aspect}")
        
        print(f"  変容準備度: {quality['transformation_readiness']:.2f}")
        
        print("\n  質的評価に基づく推奨:")
        for rec in quality['recommendations']:
            print(f"    • {rec}")
    
    # 最終状態
    print(f"\n\n最終状態:")
    print(f"  Core: {state.Λ_self_aspects['core']:.2f}")
    print(f"  Shadow: {state.Λ_self_aspects['shadow']:.2f}")
    print(f"  Social: {state.Λ_self_aspects['social']:.2f}")
    print(f"  Metacognition: {state.metacognition:.2f}")
    print(f"  Active Events: {len(state.active_events)}")
    
    return state, evaluation

# 実行例
if __name__ == "__main__":
    result_state, evaluation = run_enhanced_assessment()
