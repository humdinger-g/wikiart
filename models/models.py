import torch
from torch import nn
from timm import create_model


class ConvNext(nn.Module):
    def __init__(self, num_classes_styles, num_classes_artists, hidden=2048):
        super().__init__()
        self.model = create_model('convnextv2_large.fcmae_ft_in22k_in1k', pretrained=True)
        num_features = self.model.head.in_features
        self.model.reset_classifier(0)
        self.fc_styles = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_styles)
        )
        self.fc_artists = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_artists)
        )

    def forward(self, x):
        x = self.model(x)
        return self.fc_styles(x), self.fc_artists(x)
    


class Swin(nn.Module):
    def __init__(self, num_classes_styles, num_classes_artists, hidden=2048):
        super().__init__()
        self.model = create_model('swin_large_patch4_window12_384', pretrained=True)
        num_features = self.model.head.in_features
        self.model.reset_classifier(0)
        self.fc_styles = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_styles)
        )
        self.fc_artists = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_artists)
        )

    def forward(self, x):
        x = self.model(x)
        return self.fc_styles(x), self.fc_artists(x)
    

class SwinJoint(nn.Module):
    def __init__(self, num_classes_styles, num_classes_artists, hidden=2048):
        super().__init__()
        self.model = create_model('swin_large_patch4_window12_384', pretrained=True)
        self.num_classes_styles = num_classes_styles
        self.num_classes_artists = num_classes_artists
        num_features = self.model.head.in_features
        self.model.reset_classifier(0)
        self.fc_styles = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_styles)
        )
        self.fc_artists = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_artists)
        )

        self.fc_joint = nn.Sequential(
            nn.Linear(num_features + num_classes_styles + num_classes_artists, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes_styles * num_classes_artists)
        )

    def forward(self, x):
        x = self.model(x)
        style_logits, artist_logits = self.fc_styles(x), self.fc_artists(x)
        joint_features = torch.cat((x, style_logits, artist_logits), dim=1)
        joint_logits = self.fc_joint(joint_features).view(-1, self.num_classes_styles,
                                                           self.num_classes_artists)
        return style_logits, artist_logits, joint_logits


class StyleModel(Swin):
    def __init__(self):
        super().__init__(27, 387)
        self.fc_artists = None
    
    def forward(self, x):
        features = self.model(x)
        logits = self.fc_styles(features)
        return torch.nn.functional.softmax(logits, dim=1)


class ArtistModel(Swin):
    def __init__(self):
        super().__init__(27, 387)
        self.fc_styles = None
    
    def forward(self, x):
        features = self.model(x)
        logits = self.fc_artists(features)
        return torch.nn.functional.softmax(logits, dim=1)


def build_separate_models(model):
    style_model = StyleModel()
    artist_model = ArtistModel()
    style_model.model.load_state_dict(model.model.state_dict())
    style_model.fc_styles.load_state_dict(model.fc_styles.state_dict())

    artist_model.model.load_state_dict(model.model.state_dict())
    artist_model.fc_artists.load_state_dict(model.fc_artists.state_dict())
    return style_model, artist_model


def clip_logits(x):
    for i in range(x.size(0)):
        x[:, i][x[:, i] < x.quantile(0.8, dim=1)[i]] = -float('inf')
    return x