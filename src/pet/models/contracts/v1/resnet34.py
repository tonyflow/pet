"""Version-1 ResNet-34 backbone, classification, and segmentation contracts."""

from pet.models.manifest import (
    BackboneManifest,
    FeatureContract,
    FeatureContractExample,
    FeatureTensorSpec,
    HeadContract,
    HeadManifest,
    ModelManifest,
    contract_fingerprint,
)

RESNET34_PYRAMID_V1 = FeatureContract(
    schema_version=1,
    id="resnet34-pyramid-v1",
    kind="backbone-feature-pyramid",
    description="ResNet-34 feature pyramid before global pooling",
    input_name="image",
    input_channels=3,
    input_layout="NCHW",
    input_rank=4,
    minimum_height=32,
    minimum_width=32,
    supported_dtypes=("float16", "bfloat16", "float32"),
    color_space="RGB",
    normalization_mean=(0.485, 0.456, 0.406),
    normalization_std=(0.229, 0.224, 0.225),
    additional_keys_allowed=False,
    outputs={
        "stem": FeatureTensorSpec(64, 2, "conv1_batchnorm_relu", "low_level_edges_and_textures"),
        "layer1": FeatureTensorSpec(64, 4, "resnet_layer1", "low_level_features"),
        "layer2": FeatureTensorSpec(128, 8, "resnet_layer2", "intermediate_features"),
        "layer3": FeatureTensorSpec(256, 16, "resnet_layer3", "high_level_features"),
        "layer4": FeatureTensorSpec(512, 32, "resnet_layer4", "deepest_semantic_features"),
    },
    deterministic_given_deterministic_runtime=True,
    examples=(
        FeatureContractExample(
            (16, 3, 224, 224),
            {
                "stem": (16, 64, 112, 112),
                "layer1": (16, 64, 56, 56),
                "layer2": (16, 128, 28, 28),
                "layer3": (16, 256, 14, 14),
                "layer4": (16, 512, 7, 7),
            },
        ),
        FeatureContractExample(
            (1, 3, 256, 320),
            {
                "stem": (1, 64, 128, 160),
                "layer1": (1, 64, 64, 80),
                "layer2": (1, 128, 32, 40),
                "layer3": (1, 256, 16, 20),
                "layer4": (1, 512, 8, 10),
            },
        ),
    ),
)

CLASSIFICATION_HEAD_V1 = HeadContract(
    schema_version=1,
    id="classification-head-v1",
    kind="classification-head",
    description="Classification head over the deepest ResNet-34 feature map",
    feature_contract=RESNET34_PYRAMID_V1.id,
    consumed_keys=("layer4",),
    output_name="logits",
    output_rank=2,
    output_layout="NC",
    num_classes=37,
    spatial_policy=None,
    semantics="unnormalized_class_logits",
)

SEGMENTATION_HEAD_V1 = HeadContract(
    schema_version=1,
    id="segmentation-head-v1",
    kind="segmentation-head",
    description="Skip-connected decoder over the complete ResNet-34 feature pyramid",
    feature_contract=RESNET34_PYRAMID_V1.id,
    consumed_keys=("stem", "layer1", "layer2", "layer3", "layer4"),
    output_name="logits",
    output_rank=4,
    output_layout="NCHW",
    num_classes=2,
    spatial_policy="same_as_original_image",
    semantics="per_pixel_unnormalized_class_logits",
)

RESNET34_V1 = ModelManifest(
    schema_version=1,
    backbone=BackboneManifest(
        name="resnet34",
        version="resnet34-backbone-v1",
        feature_contract=RESNET34_PYRAMID_V1.id,
        feature_contract_definition=RESNET34_PYRAMID_V1,
        feature_contract_sha256=contract_fingerprint(RESNET34_PYRAMID_V1),
    ),
    classification=HeadManifest(
        version="classification-head-v1",
        requires_feature_contract=RESNET34_PYRAMID_V1.id,
        num_classes=37,
        interface_contract=CLASSIFICATION_HEAD_V1.id,
        interface_contract_definition=CLASSIFICATION_HEAD_V1,
        interface_contract_sha256=contract_fingerprint(CLASSIFICATION_HEAD_V1),
    ),
    segmentation=HeadManifest(
        version="segmentation-head-v1",
        requires_feature_contract=RESNET34_PYRAMID_V1.id,
        num_classes=2,
        interface_contract=SEGMENTATION_HEAD_V1.id,
        interface_contract_definition=SEGMENTATION_HEAD_V1,
        interface_contract_sha256=contract_fingerprint(SEGMENTATION_HEAD_V1),
    ),
)
