from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from enum import Enum

class SocialDomain(Enum):
    """社会的領域の分類"""
    WORK = "work"           # 職場
    FAMILY = "family"       # 家族
    PARTNER = "partner"     # 恋人/パートナー
    FRIENDS = "friends"     # 友人
    ONLINE = "online"       # ネット/SNS
    COMMUNITY = "community" # 地域/趣味のコミュニティ
    PROFESSIONAL = "professional"  # 専門家（医師、カウンセラー等）

@dataclass
class DomainSpecificSocialState:
    """領域特異的な社会的状態"""
    domain: SocialDomain
    resonance_rate: float = 0.5      # 共鳴率（0-1: どれだけ本音で繋がれるか）
    shadow_eruption: float = 0.3     # Shadow噴出度（0-1: 抑圧された側面の表出）
    support_quality: float = 0.5     # サポート構造の質（0-1: 実質的な支援）
    tension_level: float = 0.5       # 緊張度（0-1: ストレスレベル）
    energy_balance: float = 0.0      # エネルギー収支（-1〜1: 奪われる〜与えられる）
    authenticity: float = 0.5        # 真正性（0-1: どれだけ本来の自分でいられるか）
    safety_index: float = 0.5        # 安全指数（0-1: 心理的安全性）
    
    def calculate_domain_health(self) -> float:
        """この領域の健全性スコアを計算"""
        # 正の要因
        positive = (
            self.resonance_rate * 0.3 +
            self.support_quality * 0.3 +
            self.authenticity * 0.2 +
            self.safety_index * 0.2
        )
        
        # 負の要因
        negative = (
            self.shadow_eruption * 0.3 +
            self.tension_level * 0.4 +
            max(0, -self.energy_balance) * 0.3
        )
        
        return np.clip(positive - negative * 0.5, 0, 1)
    
    def is_resource_domain(self) -> bool:
        """この領域がリソース（エネルギー源）かどうか"""
        return self.energy_balance > 0.3 and self.safety_index > 0.6
    
    def is_drain_domain(self) -> bool:
        """この領域がエネルギーを奪うかどうか"""
        return self.energy_balance < -0.3 or self.tension_level > 0.7

@dataclass 
class MultidimensionalSocialSelf:
    """多次元社会的自己"""
    domains: Dict[SocialDomain, DomainSpecificSocialState] = field(default_factory=dict)
    inter_domain_effects: Dict[tuple, float] = field(default_factory=dict)  # 領域間相互作用
    
    def __post_init__(self):
        # デフォルトで主要領域を初期化
        if not self.domains:
            for domain in [SocialDomain.WORK, SocialDomain.FAMILY, 
                          SocialDomain.FRIENDS, SocialDomain.ONLINE]:
                self.domains[domain] = DomainSpecificSocialState(domain=domain)
    
    def add_feedback(self, domain: SocialDomain, feedback: 'SocialFeedback'):
        """特定領域へのフィードバックを処理"""
        if domain not in self.domains:
            self.domains[domain] = DomainSpecificSocialState(domain=domain)
        
        state = self.domains[domain]
        
        # フィードバックの影響を計算
        impact = feedback.valence * feedback.intensity * feedback.authenticity
        
        # 各パラメータへの影響
        if impact > 0:
            state.resonance_rate = min(1.0, state.resonance_rate + impact * 0.1)
            state.support_quality = min(1.0, state.support_quality + impact * 0.08)
            state.energy_balance = min(1.0, state.energy_balance + impact * 0.15)
            state.safety_index = min(1.0, state.safety_index + impact * 0.05)
        else:
            state.tension_level = min(1.0, state.tension_level - impact * 0.1)
            state.energy_balance = max(-1.0, state.energy_balance + impact * 0.15)
            state.shadow_eruption = min(1.0, state.shadow_eruption - impact * 0.05)
        
        # 真正性は複雑な更新
        if feedback.category in ['authentic_connection', 'deep_sharing']:
            state.authenticity = min(1.0, state.authenticity + 0.1)
        elif feedback.category in ['facade_maintenance', 'role_playing']:
            state.authenticity = max(0.0, state.authenticity - 0.1)
    
    def calculate_resource_distribution(self) -> Dict[str, any]:
        """リソースとドレインの分布を分析"""
        resources = []
        drains = []
        neutral = []
        
        for domain, state in self.domains.items():
            if state.is_resource_domain():
                resources.append({
                    'domain': domain.value,
                    'strength': state.energy_balance,
                    'quality': state.support_quality
                })
            elif state.is_drain_domain():
                drains.append({
                    'domain': domain.value,
                    'severity': -state.energy_balance,
                    'tension': state.tension_level
                })
            else:
                neutral.append(domain.value)
        
        # 全体的なバランス
        total_energy = sum(state.energy_balance for state in self.domains.values())
        
        return {
            'resources': resources,
            'drains': drains,
            'neutral': neutral,
            'total_energy_balance': total_energy,
            'is_sustainable': total_energy > -0.5 and len(resources) > 0
        }
    
    def identify_isolation_patterns(self) -> List[Dict]:
        """孤立パターンの特定"""
        isolation_indicators = []
        
        for domain, state in self.domains.items():
            isolation_score = 0.0
            
            # 低共鳴率
            if state.resonance_rate < 0.3:
                isolation_score += 0.3
            
            # 低サポート
            if state.support_quality < 0.3:
                isolation_score += 0.3
            
            # 高Shadow（本音を隠している）
            if state.shadow_eruption > 0.7:
                isolation_score += 0.2
            
            # 低真正性
            if state.authenticity < 0.3:
                isolation_score += 0.2
            
            if isolation_score > 0.5:
                isolation_indicators.append({
                    'domain': domain.value,
                    'isolation_score': isolation_score,
                    'primary_issue': self._identify_primary_issue(state)
                })
        
        return isolation_indicators
    
    def _identify_primary_issue(self, state: DomainSpecificSocialState) -> str:
        """主要な問題を特定"""
        issues = []
        
        if state.resonance_rate < 0.3:
            issues.append(('low_resonance', 0.3 - state.resonance_rate))
        if state.authenticity < 0.3:
            issues.append(('low_authenticity', 0.3 - state.authenticity))
        if state.tension_level > 0.7:
            issues.append(('high_tension', state.tension_level - 0.7))
        if state.shadow_eruption > 0.7:
            issues.append(('shadow_overactive', state.shadow_eruption - 0.7))
        
        if issues:
            return max(issues, key=lambda x: x[1])[0]
        return 'general_disconnection'
    
    def calculate_spillover_effects(self) -> Dict[str, List[str]]:
        """領域間のスピルオーバー効果を計算"""
        spillovers = {}
        
        # 職場ストレスの家族への転移
        if (SocialDomain.WORK in self.domains and 
            SocialDomain.FAMILY in self.domains):
            work = self.domains[SocialDomain.WORK]
            if work.tension_level > 0.7 or work.energy_balance < -0.5:
                spillovers['work_to_family'] = [
                    'irritability_at_home',
                    'emotional_unavailability',
                    'displaced_frustration'
                ]
        
        # オンラインでのShadow噴出
        if SocialDomain.ONLINE in self.domains:
            online = self.domains[SocialDomain.ONLINE]
            if online.shadow_eruption > 0.6:
                spillovers['shadow_online'] = [
                    'anonymous_aggression',
                    'idealized_self_presentation',
                    'validation_seeking'
                ]
        
        # 家族問題の友人関係への影響
        if (SocialDomain.FAMILY in self.domains and 
            SocialDomain.FRIENDS in self.domains):
            family = self.domains[SocialDomain.FAMILY]
            if family.tension_level > 0.6:
                spillovers['family_to_friends'] = [
                    'emotional_dumping',
                    'avoidance_patterns',
                    'trust_issues'
                ]
        
        return spillovers
    
    def generate_intervention_map(self) -> Dict[str, any]:
        """介入マップの生成"""
        distribution = self.calculate_resource_distribution()
        isolation = self.identify_isolation_patterns()
        spillovers = self.calculate_spillover_effects()
        
        interventions = {
            'immediate_priorities': [],
            'resource_reinforcement': [],
            'boundary_work': [],
            'integration_opportunities': []
        }
        
        # 即時介入が必要な領域
        for drain in distribution['drains']:
            if drain['severity'] > 0.7:
                interventions['immediate_priorities'].append({
                    'domain': drain['domain'],
                    'action': 'urgent_support_or_boundary_setting',
                    'specific_steps': self._get_domain_specific_interventions(drain['domain'])
                })
        
        # リソース強化
        for resource in distribution['resources']:
            interventions['resource_reinforcement'].append({
                'domain': resource['domain'],
                'action': 'protect_and_nurture',
                'strategies': [
                    'schedule_regular_connection_time',
                    'express_gratitude',
                    'deepen_authentic_sharing'
                ]
            })
        
        # 境界設定が必要な領域
        for domain, state in self.domains.items():
            if state.energy_balance < -0.5 and state.authenticity < 0.4:
                interventions['boundary_work'].append({
                    'domain': domain.value,
                    'action': 'establish_healthy_boundaries',
                    'focus': 'protect_authenticity_and_energy'
                })
        
        # 統合の機会
        if len(distribution['resources']) > 0 and len(isolation) > 0:
            interventions['integration_opportunities'].append({
                'strategy': 'bridge_isolated_and_resource_domains',
                'example': f"Leverage {distribution['resources'][0]['domain']} support to address {isolation[0]['domain']} isolation"
            })
        
        return interventions
    
    def _get_domain_specific_interventions(self, domain: str) -> List[str]:
        """領域特異的な介入策"""
        interventions_map = {
            'work': [
                'Set clear work-life boundaries',
                'Seek supervisor support or HR consultation',
                'Practice assertive communication',
                'Consider role or team changes'
            ],
            'family': [
                'Family therapy or counseling',
                'Establish personal space and time',
                'Open communication about needs',
                'Create new positive rituals'
            ],
            'partner': [
                'Couples counseling',
                'Regular check-ins about relationship',
                'Individual therapy for attachment issues',
                'Planned quality time without distractions'
            ],
            'friends': [
                'Evaluate friendship dynamics',
                'Set boundaries with draining friends',
                'Seek new connections with shared values',
                'Practice vulnerability in safe relationships'
            ],
            'online': [
                'Digital detox periods',
                'Curate online connections mindfully',
                'Set time limits for social media',
                'Engage in meaningful online communities'
            ]
        }
        
        return interventions_map.get(domain, ['Seek professional guidance'])

class DomainQuestion:
    """領域別質問"""
    def __init__(self, id: str, domain: SocialDomain, parameter: str, 
                 text: str, scale: str, weight: float, reverse_scored: bool = False):
        self.id = id
        self.domain = domain
        self.parameter = parameter
        self.text = text
        self.scale = scale
        self.weight = weight
        self.reverse_scored = reverse_scored

class MultidimensionalSocialAssessment:
    """多次元社会的自己のアセスメント質問システム"""
    
    def __init__(self):
        self.questions = self._generate_all_questions()
    
    def _generate_all_questions(self) -> Dict[SocialDomain, List[DomainQuestion]]:
        """全領域の質問を生成"""
        questions = {}
        for domain in SocialDomain:
            questions[domain] = self._generate_domain_questions(domain)
        return questions
    
    def _generate_domain_questions(self, domain: SocialDomain) -> List[DomainQuestion]:
        """特定領域の質問群を生成"""
        domain_names = {
            SocialDomain.WORK: "職場",
            SocialDomain.FAMILY: "家族",
            SocialDomain.PARTNER: "恋人・パートナー",
            SocialDomain.FRIENDS: "友人",
            SocialDomain.ONLINE: "ネット・SNS",
            SocialDomain.COMMUNITY: "趣味・地域のコミュニティ",
            SocialDomain.PROFESSIONAL: "専門家（医師・カウンセラー等）"
        }
        
        domain_name = domain_names[domain]
        domain_id = domain.value
        
        questions = [
            # 共鳴率（本音度）
            DomainQuestion(
                id=f"{domain_id}_resonance_1",
                domain=domain,
                parameter="resonance_rate",
                text=f"{domain_name}では、自分の本音や本当の気持ちを表現できていますか？",
                scale="1:全くできない 2:あまりできない 3:時々できる 4:だいたいできる 5:いつもできる",
                weight=1.0
            ),
            # Shadow噴出度
            DomainQuestion(
                id=f"{domain_id}_shadow_1",
                domain=domain,
                parameter="shadow_eruption",
                text=f"{domain_name}では、普段抑えている感情や衝動が突然出てしまうことがありますか？",
                scale="1:全くない 2:めったにない 3:時々ある 4:よくある 5:頻繁にある",
                weight=1.0
            ),
            # サポート構造
            DomainQuestion(
                id=f"{domain_id}_support_1",
                domain=domain,
                parameter="support_quality",
                text=f"{domain_name}で困った時、実際に助けてくれる人はいますか？",
                scale="1:誰もいない 2:1人くらい 3:2-3人 4:数人いる 5:たくさんいる",
                weight=1.0
            ),
            # 緊張度
            DomainQuestion(
                id=f"{domain_id}_tension_1",
                domain=domain,
                parameter="tension_level",
                text=f"{domain_name}にいる時、緊張や不安を感じますか？",
                scale="1:全く感じない 2:あまり感じない 3:時々感じる 4:よく感じる 5:常に感じる",
                weight=1.0
            ),
            # エネルギー収支
            DomainQuestion(
                id=f"{domain_id}_energy_1",
                domain=domain,
                parameter="energy_balance",
                text=f"{domain_name}での時間を過ごした後、エネルギーが充電される感じがしますか、それとも消耗しますか？",
                scale="1:とても充電される 2:やや充電される 3:変わらない 4:やや消耗する 5:とても消耗する",
                weight=1.0,
                reverse_scored=True
            ),
            # 真正性
            DomainQuestion(
                id=f"{domain_id}_authenticity_1",
                domain=domain,
                parameter="authenticity",
                text=f"{domain_name}では、「本来の自分」でいられていますか？",
                scale="1:全くいられない 2:あまりいられない 3:半分くらい 4:だいたいいられる 5:完全にいられる",
                weight=1.0
            ),
            # 安全指数
            DomainQuestion(
                id=f"{domain_id}_safety_1",
                domain=domain,
                parameter="safety_index",
                text=f"{domain_name}は、心理的に安全だと感じられる場所ですか？",
                scale="1:全く安全でない 2:あまり安全でない 3:まあまあ安全 4:かなり安全 5:とても安全",
                weight=1.0
            )
        ]
        
        return questions
    
    def calculate_domain_scores(self, responses: Dict[str, int]) -> Dict[SocialDomain, Dict[str, float]]:
        """回答から各領域のスコアを計算"""
        domain_scores = {}
        
        for domain, questions in self.questions.items():
            parameter_scores = {param: [] for param in [
                "resonance_rate", "shadow_eruption", "support_quality",
                "tension_level", "energy_balance", "authenticity", "safety_index"
            ]}
            
            for q in questions:
                if q.id in responses:
                    score = responses[q.id]
                    if q.reverse_scored:
                        score = 6 - score
                    normalized_score = (score - 1) / 4
                    parameter_scores[q.parameter].append(normalized_score * q.weight)
            
            domain_scores[domain] = {}
            for param, scores in parameter_scores.items():
                if scores:
                    domain_scores[domain][param] = sum(scores) / len(scores)
                else:
                    domain_scores[domain][param] = 0.5
        
        return domain_scores

class EnhancedSocialDynamics:
    """拡張版：社会的ダイナミクスの統合（質問システム付き）"""
    
    def __init__(self):
        self.social_self = MultidimensionalSocialSelf()
        self.assessment = MultidimensionalSocialAssessment()
    
    def visualize_social_topology(self) -> Dict[str, any]:
        """社会的位相図の可視化データ生成"""
        topology = {
            'nodes': [],
            'edges': [],
            'clusters': {
                'resources': [],
                'drains': [],
                'neutral': []
            }
        }
        
        # 各領域をノードとして配置
        for domain, state in self.social_self.domains.items():
            node = {
                'id': domain.value,
                'label': domain.value.capitalize(),
                'size': abs(state.energy_balance) * 50 + 20,  # エネルギーの絶対値でサイズ
                'color': self._get_node_color(state),
                'x': state.resonance_rate * 100,  # 共鳴率でX座標
                'y': state.authenticity * 100,     # 真正性でY座標
                'health': state.calculate_domain_health()
            }
            topology['nodes'].append(node)
            
            # クラスター分類
            if state.is_resource_domain():
                topology['clusters']['resources'].append(domain.value)
            elif state.is_drain_domain():
                topology['clusters']['drains'].append(domain.value)
            else:
                topology['clusters']['neutral'].append(domain.value)
        
        # 領域間の相互作用をエッジとして
        spillovers = self.social_self.calculate_spillover_effects()
        for effect_type, effects in spillovers.items():
            parts = effect_type.split('_to_')
            if len(parts) == 2:
                topology['edges'].append({
                    'source': parts[0],
                    'target': parts[1],
                    'type': 'spillover',
                    'weight': len(effects)
                })
        
        return topology
    
    def _get_node_color(self, state: DomainSpecificSocialState) -> str:
        """ノードの色を状態に基づいて決定"""
        if state.is_resource_domain():
            return '#4CAF50'  # 緑：エネルギー源
        elif state.is_drain_domain():
            return '#F44336'  # 赤：エネルギードレイン
        elif state.calculate_domain_health() > 0.6:
            return '#2196F3'  # 青：健全
        else:
            return '#FF9800'  # オレンジ：要注意
    
    def generate_detailed_assessment(self) -> Dict[str, any]:
        """詳細なアセスメント生成"""
        distribution = self.social_self.calculate_resource_distribution()
        isolation = self.social_self.identify_isolation_patterns()
        spillovers = self.social_self.calculate_spillover_effects()
        interventions = self.social_self.generate_intervention_map()
        topology = self.visualize_social_topology()
        
        # 全体的な健康度
        overall_health = np.mean([
            state.calculate_domain_health() 
            for state in self.social_self.domains.values()
        ])
        
        # 最も問題のある領域
        problematic_domains = sorted(
            self.social_self.domains.items(),
            key=lambda x: x[1].calculate_domain_health()
        )[:3]
        
        # 最も支援的な領域
        supportive_domains = sorted(
            self.social_self.domains.items(),
            key=lambda x: x[1].calculate_domain_health(),
            reverse=True
        )[:3]
        
        return {
            'overall_social_health': overall_health,
            'energy_distribution': distribution,
            'isolation_patterns': isolation,
            'spillover_effects': spillovers,
            'intervention_map': interventions,
            'topology_data': topology,
            'critical_domains': [
                {
                    'domain': d[0].value,
                    'health': d[1].calculate_domain_health(),
                    'main_issues': self.social_self._identify_primary_issue(d[1])
                }
                for d in problematic_domains
            ],
            'strength_domains': [
                {
                    'domain': d[0].value,
                    'health': d[1].calculate_domain_health(),
                    'key_resources': {
                        'resonance': d[1].resonance_rate,
                        'support': d[1].support_quality,
                        'safety': d[1].safety_index
                    }
                }
                for d in supportive_domains
            ],
            'recommendations': self._generate_holistic_recommendations(
                overall_health, distribution, isolation, spillovers
            )
        }
    
    def _generate_holistic_recommendations(self, overall_health: float,
                                         distribution: Dict,
                                         isolation: List,
                                         spillovers: Dict) -> List[str]:
        """統合的な推奨事項の生成"""
        recommendations = []
        
        # 全体的な健康度に基づく
        if overall_health < 0.4:
            recommendations.append(
                "社会的サポートシステム全体の再構築が必要。専門家の支援を検討"
            )
        
        # エネルギーバランスに基づく
        if not distribution['is_sustainable']:
            recommendations.append(
                "エネルギー収支が負。ドレイン領域での境界設定が急務"
            )
        
        # 孤立パターンに基づく
        if len(isolation) > 2:
            recommendations.append(
                "複数領域で孤立傾向。安全な領域から徐々に本音の共有を開始"
            )
        
        # スピルオーバーに基づく
        if len(spillovers) > 1:
            recommendations.append(
                "領域間の負の転移が発生。各領域での感情処理方法の学習が必要"
            )
        
        # リソース活用
        if len(distribution['resources']) > 0:
            resource_domain = distribution['resources'][0]['domain']
            recommendations.append(
                f"{resource_domain}での肯定的体験を他領域にも応用することを検討"
            )
        
        return recommendations
    
    def assess_from_responses(self, responses: Dict[str, int]) -> Dict[str, any]:
        """質問への回答から完全な分析を実行"""
        
        # 1. 回答からスコアを計算
        domain_scores = self.assessment.calculate_domain_scores(responses)
        
        # 2. スコアを多次元社会的自己モデルに適用
        for domain, scores in domain_scores.items():
            if domain in self.social_self.domains:
                state = self.social_self.domains[domain]
                # 各パラメータを更新
                state.resonance_rate = scores.get('resonance_rate', 0.5)
                state.shadow_eruption = scores.get('shadow_eruption', 0.5)
                state.support_quality = scores.get('support_quality', 0.5)
                state.tension_level = scores.get('tension_level', 0.5)
                state.authenticity = scores.get('authenticity', 0.5)
                state.safety_index = scores.get('safety_index', 0.5)
                
                # エネルギーバランスの計算（-1〜1の範囲に変換）
                energy_score = scores.get('energy_balance', 0.5)
                state.energy_balance = (energy_score - 0.5) * 2
        
        # 3. 詳細な分析を実行
        return self.generate_detailed_assessment()
    
    def generate_question_based_report(self, responses: Dict[str, int]) -> str:
        """質問ベースの詳細レポート生成"""
        
        # アセスメント実行
        assessment = self.assess_from_responses(responses)
        
        report = []
        report.append("=== 多次元社会的自己 総合評価レポート ===\n")
        
        # 1. 全体サマリー
        report.append(f"【全体的な社会的健康度】")
        health = assessment['overall_social_health']
        if health > 0.7:
            status = "良好"
        elif health > 0.5:
            status = "やや課題あり"
        elif health > 0.3:
            status = "要注意"
        else:
            status = "危機的"
        report.append(f"スコア: {health:.2f} ({status})\n")
        
        # 2. エネルギー分析
        report.append(f"【エネルギー収支分析】")
        energy_dist = assessment['energy_distribution']
        report.append(f"・エネルギー源となる領域: {[r['domain'] for r in energy_dist['resources']]}")
        report.append(f"・エネルギーを消耗する領域: {[d['domain'] for d in energy_dist['drains']]}")
        report.append(f"・持続可能性: {'○' if energy_dist['is_sustainable'] else '×'}\n")
        
        # 3. 領域別詳細
        report.append(f"【領域別状態】")
        for domain in self.social_self.domains.values():
            health = domain.calculate_domain_health()
            if health < 0.4:  # 問題のある領域のみ詳細表示
                report.append(f"\n《{domain.domain.value}》 健康度: {health:.2f}")
                report.append(f"  主な問題:")
                if domain.resonance_rate < 0.3:
                    report.append(f"  - 本音を表現できない（共鳴率: {domain.resonance_rate:.2f}）")
                if domain.shadow_eruption > 0.7:
                    report.append(f"  - 抑圧された感情の噴出（Shadow: {domain.shadow_eruption:.2f}）")
                if domain.energy_balance < -0.5:
                    report.append(f"  - 深刻なエネルギー消耗（収支: {domain.energy_balance:.2f}）")
                if domain.authenticity < 0.3:
                    report.append(f"  - 本来の自分を失っている（真正性: {domain.authenticity:.2f}）")
        
        # 4. 推奨事項
        report.append(f"\n【推奨される対応】")
        for i, rec in enumerate(assessment['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        return "\n".join(report)

# 統合的な使用例
def demonstrate_integrated_social_assessment():
    """多次元社会的自己の分析デモ"""
    
    # 社会的ダイナミクスの初期化
    dynamics = EnhancedSocialDynamics()
    
    # 各領域の状態を設定（例：職場で疲弊、家族は安定、ネットでShadow噴出）
    
    # 職場：高ストレス、低サポート
    dynamics.social_self.domains[SocialDomain.WORK] = DomainSpecificSocialState(
        domain=SocialDomain.WORK,
        resonance_rate=0.2,    # 本音を言えない
        shadow_eruption=0.8,   # 抑圧が強い
        support_quality=0.2,   # サポート不足
        tension_level=0.9,     # 高ストレス
        energy_balance=-0.8,   # エネルギーを奪われる
        authenticity=0.1,      # 仮面をかぶっている
        safety_index=0.2       # 心理的に危険
    )
    
    # 家族：比較的安定
    dynamics.social_self.domains[SocialDomain.FAMILY] = DomainSpecificSocialState(
        domain=SocialDomain.FAMILY,
        resonance_rate=0.7,
        shadow_eruption=0.3,
        support_quality=0.7,
        tension_level=0.4,
        energy_balance=0.5,
        authenticity=0.6,
        safety_index=0.8
    )
    
    # オンライン：Shadow噴出
    dynamics.social_self.domains[SocialDomain.ONLINE] = DomainSpecificSocialState(
        domain=SocialDomain.ONLINE,
        resonance_rate=0.4,
        shadow_eruption=0.9,   # 匿名で攻撃的
        support_quality=0.2,
        tension_level=0.6,
        energy_balance=-0.3,
        authenticity=0.3,      # 別人格
        safety_index=0.4
    )
    
    # 友人：表面的
    dynamics.social_self.domains[SocialDomain.FRIENDS] = DomainSpecificSocialState(
        domain=SocialDomain.FRIENDS,
        resonance_rate=0.4,
        shadow_eruption=0.5,
        support_quality=0.4,
        tension_level=0.5,
        energy_balance=0.1,
        authenticity=0.4,
        safety_index=0.5
    )
    
    # 詳細なアセスメント実行
    assessment = dynamics.generate_detailed_assessment()
    
    # 結果の表示
    print("=== 多次元社会的自己アセスメント ===\n")
    
    print(f"全体的な社会的健康度: {assessment['overall_social_health']:.2f}\n")
    
    print("【エネルギー分布】")
    print(f"  リソース領域: {[r['domain'] for r in assessment['energy_distribution']['resources']]}")
    print(f"  ドレイン領域: {[d['domain'] for d in assessment['energy_distribution']['drains']]}")
    print(f"  持続可能性: {'Yes' if assessment['energy_distribution']['is_sustainable'] else 'No'}")
    
    print("\n【孤立パターン】")
    for isolation in assessment['isolation_patterns']:
        print(f"  {isolation['domain']}: 孤立スコア {isolation['isolation_score']:.2f} "
              f"(主要問題: {isolation['primary_issue']})")
    
    print("\n【領域間スピルオーバー】")
    for effect_type, effects in assessment['spillover_effects'].items():
        print(f"  {effect_type}: {', '.join(effects)}")
    
    print("\n【緊急介入が必要な領域】")
    for priority in assessment['intervention_map']['immediate_priorities']:
        print(f"  {priority['domain']}: {priority['action']}")
        for step in priority['specific_steps']:
            print(f"    - {step}")
    
    print("\n【強みとなる領域】")
    for strength in assessment['strength_domains']:
        print(f"  {strength['domain']}: 健康度 {strength['health']:.2f}")
        print(f"    共鳴率: {strength['key_resources']['resonance']:.2f}, "
              f"サポート: {strength['key_resources']['support']:.2f}, "
              f"安全性: {strength['key_resources']['safety']:.2f}")
    
    print("\n【統合的推奨事項】")
    for i, rec in enumerate(assessment['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    return dynamics, assessment

# 実行
if __name__ == "__main__":
    dynamics, assessment = demonstrate_multidimensional_social_analysis()
