# gap-conf-kaggle
Kaggle の Gender Pronoun Resolutionコンペの実装

## URLs
- [データセットのgit](https://github.com/google-research-datasets/gap-coreference)
- [コンペの説明](https://www.kaggle.com/c/gendered-pronoun-resolution)

## コンペデータ情報
- offsetは参照テキストの文字レベルでの指定文字の開始位置のオフセット（配列のインデックス）

## BASELINE
- https://arxiv.org/pdf/1810.05201.pdf のTable6を参照。
ランダム、トークン間の距離、最も近い候補（複数ある場合）、統語構造の類似性、統語構造と最も近いやつのどちらかを依存ラベルにより判定。ヒューリスティックによるアプローチで最後の方法がよく70%ほどの正解率
- BASELINEおよび既存ツールのエラーパターンは、以下
    - 人名そのものではなく、そのロールにラベルが付いているパターン
    - 統語構造が複雑な場合
    - 指示しているものが、歴史的な用語である場合
    - ドメイン特有の知識が必要な場合
    - アノテーションエラー
