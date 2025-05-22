import argparse
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from alive_progress import alive_bar

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(0)
torch.cuda.manual_seed(0)

np.set_printoptions(linewidth=np.inf)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="./data/cvpr_24.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.6,
        help="Mean similarity threshold for related papers (0.0-1.0)",
    )
    parser.add_argument(
        "--top_k_similar",
        type=int,
        default=3,
        help="Number of top reference papers to consider for similarity",
    )
    parser.add_argument(
        "--min_top_papers",
        type=int,
        default=2,
        help="Minimum number of papers that must exceed individual similarity threshold",
    )
    parser.add_argument(
        "--individual_sim_threshold",
        type=float,
        default=0.65,
        help="Threshold for individual paper similarity",
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=15.0,
        help="Select papers in the top percentile of similarity scores",
    )
    return parser.parse_args()


def load_model() -> SentenceTransformer:
    return SentenceTransformer('intfloat/multilingual-e5-large-instruct')


def load_data(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    datas = pd.read_csv(file_path)
    datas = datas.rename(columns={"Title": "title", "Abstract": "abstract"})
    titles = datas["title"].tolist()
    abstracts = datas["abstract"].tolist()
    paper_list = [
        title + ": " + abstract
        for title, abstract in zip(datas["title"], datas["abstract"])
    ]
    return titles, abstracts, paper_list


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[a-z]+{([^}]*)}", r"\1", text)
    text = re.sub(r"https?://\S+", "URL", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_embeddings(
    texts: List[str], model: SentenceTransformer, batch_size: int
) -> np.ndarray:
    embeddings = []
    with alive_bar(len(range(0, len(texts), batch_size)), title="임베딩 계산 중") as bar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = model.encode(
                batch, convert_to_tensor=True, normalize_embeddings=True
            )
            embeddings.append(batch_embeddings.cpu().numpy())
            bar()
    return np.concatenate(embeddings, axis=0)


def is_paper_related(
    similarities: np.ndarray, args: argparse.Namespace
) -> Tuple[bool, str]:
    # 1. 평균 유사도 체크
    mean_sim = np.mean(similarities)
    if mean_sim > args.similarity_threshold:
        return True, f"평균 유사도: {mean_sim:.4f} > {args.similarity_threshold}"

    # 2. 상위 k개 중 개별 threshold를 넘는 논문 개수 체크
    top_sims = np.sort(similarities)[-args.top_k_similar :]
    num_above_threshold = sum(sim >= args.individual_sim_threshold for sim in top_sims)
    if num_above_threshold >= args.min_top_papers:
        return (
            True,
            f"상위 {args.top_k_similar}개 논문 중 {num_above_threshold}개가 개별 유사도 threshold {args.individual_sim_threshold} 초과",
        )

    return False, ""


def process_papers(
    titles: List[str],
    abstracts: List[str],
    paper_embeddings: np.ndarray,
    compare_papers_embeddings: np.ndarray,
    compare_papers: Dict[str, str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], int]:
    related_papers = []
    num_paper = 0
    compare_papers_embeddings_tensor = torch.tensor(compare_papers_embeddings)

    with alive_bar(len(paper_embeddings), title="논문 분석 중") as bar:
        for i, paper_embedding_np in enumerate(paper_embeddings):
            title = titles[i]
            abstract = abstracts[i]

            paper_embedding_tensor = torch.tensor(paper_embedding_np).unsqueeze(0)
            similarities_tensor = util.pytorch_cos_sim(compare_papers_embeddings_tensor, paper_embedding_tensor)
            abstract_similarity = similarities_tensor.cpu().numpy().flatten()

            mean_similarity = np.mean(abstract_similarity)

            is_related, reason = is_paper_related(abstract_similarity, args)

            if is_related:
                detailed_similarities = {}
                for comp_title, sim in zip(compare_papers.keys(), abstract_similarity):
                    detailed_similarities[comp_title] = float(sim)

                sorted_similarities = {
                    k: v
                    for k, v in sorted(
                        detailed_similarities.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                }
                related_papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "abstract_similarity": abstract_similarity,
                        "mean_abstract_similarity": mean_similarity,
                        "detailed_similarities": sorted_similarities,
                    }
                )

                print(f"\n{'=' * 80}")
                print(f"관련 논문 #{num_paper + 1}: {title}")
                print(f"평균 유사도: {mean_similarity:.4f}")
                print(f"선정 이유: {reason}")
                print("\n요약:")
                print(abstract[:100] + "...")
                print("\n가장 유사한 논문 TOP 3:")
                for idx, (comp_title, sim) in enumerate(
                    list(sorted_similarities.items())[:3]
                ):
                    print(f"  {idx + 1}. {comp_title} (유사도: {sim:.4f})")
                print(f"{'=' * 80}")

                num_paper += 1
            bar()

    return related_papers, num_paper


def save_results(
    related_papers: List[Dict[str, Any]], file_path: str
) -> Tuple[str, str]:
    related_paper_df = pd.DataFrame(
        [
            {
                "title": paper["title"],
                "abstract": paper["abstract"],
                "mean_abstract_similarity": paper["mean_abstract_similarity"],
                "detailed_similarities": paper["detailed_similarities"],
            }
            for paper in related_papers
        ]
    )

    related_paper_df = related_paper_df.sort_values(
        "mean_abstract_similarity", ascending=False
    )
    output_file = file_path.replace(".csv", "-related_papers.csv")
    related_paper_df[["title", "abstract", "mean_abstract_similarity"]].to_csv(
        output_file, index=False, mode="w"
    )
    json_output_file = file_path.replace(".csv", "-related_papers_detailed.json")
    related_paper_df.to_json(json_output_file, orient="records", indent=4)

    return output_file, json_output_file


def main():
    args = parse_arguments()
    model = load_model()
    titles, abstracts, _ = load_data(args.file)

    print("기준 논문과 입력 논문 임베딩 계산을 시작합니다.")
    compare_papers = {
        "Ablating Concepts in Text-to-Image Diffusion Models": "Large-scale text-to-image diffusion models can generate high-fidelity images with powerful compositional ability. However, these models are typically trained on an enormous amount of Internet data, often containing copyrighted material, licensed images, and personal photos. Furthermore, they have been found to replicate the style of various living artists or memorize exact training samples. How can we remove such copyrighted concepts or images without retraining the model from scratch? To achieve this goal, we propose an efficient method of ablating concepts in the pretrained model, i.e., preventing the generation of a target concept. Our algorithm learns to match the image distribution for a target style, instance, or text prompt we wish to ablate to the distribution corresponding to an anchor concept. This prevents the model from generating target concepts given its text condition. Extensive experiments show that our method can successfully prevent the generation of the ablated concept while preserving closely related concepts in the model.",
        "Erasing Concepts from Diffusion Models": "Motivated by recent advancements in text-to-image diffusion, we study erasure of specific concepts from the model's weights. While Stable Diffusion has shown promise in producing explicit or realistic artwork, it has raised concerns regarding its potential for misuse. We propose a fine-tuning method that can erase a visual concept from a pre-trained diffusion model, given only the name of the style and using negative guidance as a teacher. We benchmark our method against previous approaches that remove sexually explicit content and demonstrate its effectiveness, performing on par with Safe Latent Diffusion and censored training. To evaluate artistic style removal, we conduct experiments erasing five modern artists from the network and conduct a user study to assess the human perception of the removed styles. Unlike previous methods, our approach can remove concepts from a diffusion model permanently rather than modifying the output at the inference time, so it cannot be circumvented even if a user has access to model weights. Our code, data, and results are available at this https URL",
        "Forget-Me-Not: Learning to Forget in Text-to-Image Diffusion Models": r"The unlearning problem of deep learning models, once primarily an academic concern, has become a prevalent issue in the industry. The significant advances in text-to-image generation techniques have prompted global discussions on privacy, copyright, and safety, as numerous unauthorized personal IDs, content, artistic creations, and potentially harmful materials have been learned by these models and later utilized to generate and distribute uncontrolled content. To address this challenge, we propose \textbf{Forget-Me-Not}, an efficient and low-cost solution designed to safely remove specified IDs, objects, or styles from a well-configured text-to-image model in as little as 30 seconds, without impairing its ability to generate other content. Alongside our method, we introduce the \textbf{Memorization Score (M-Score)} and \textbf{ConceptBench} to measure the models' capacity to generate general concepts, grouped into three primary categories: ID, object, and style. Using M-Score and ConceptBench, we demonstrate that Forget-Me-Not can effectively eliminate targeted concepts while maintaining the model's performance on other concepts. Furthermore, Forget-Me-Not offers two practical extensions: a) removal of potentially harmful or NSFW content, and b) enhancement of model accuracy, inclusion and diversity through \textbf{concept correction and disentanglement}. It can also be adapted as a lightweight model patch for Stable Diffusion, allowing for concept manipulation and convenient distribution. To encourage future research in this critical area and promote the development of safe and inclusive generative models, we will open-source our code and ConceptBench at \href{this https URL}{this https URL}.",
        "MACE: Mass Concept Erasure in Diffusion Models": "The rapid expansion of large-scale text-to-image diffusion models has raised growing concerns regarding their potential misuse in creating harmful or misleading content. In this paper, we introduce MACE, a finetuning framework for the task of mass concept erasure. This task aims to prevent models from generating images that embody unwanted concepts when prompted. Existing concept erasure methods are typically restricted to handling fewer than five concepts simultaneously and struggle to find a balance between erasing concept synonyms (generality) and maintaining unrelated concepts (specificity). In contrast, MACE differs by successfully scaling the erasure scope up to 100 concepts and by achieving an effective balance between generality and specificity. This is achieved by leveraging closed-form cross-attention refinement along with LoRA finetuning, collectively eliminating the information of undesirable concepts. Furthermore, MACE integrates multiple LoRAs without mutual interference. We conduct extensive evaluations of MACE against prior methods across four different tasks: object erasure, celebrity erasure, explicit content erasure, and artistic style erasure. Our results reveal that MACE surpasses prior methods in all evaluated tasks. Code is available at https://github.com/Shilin-LU/MACE.",
        "Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models": "The recent proliferation of large-scale text-to-image models has led to growing concerns that such models may be misused to generate harmful, misleading, and inappropriate content. Motivated by this issue, we derive a technique inspired by continual learning to selectively forget concepts in pretrained deep generative models. Our method, dubbed Selective Amnesia, enables controllable forgetting where a user can specify how a concept should be forgotten. Selective Amnesia can be applied to conditional variational likelihood models, which encompass a variety of popular deep generative frameworks, including variational autoencoders and large-scale text-to-image diffusion models. Experiments across different models demonstrate that our approach induces forgetting on a variety of concepts, from entire classes in standard datasets to celebrity and nudity prompts in text-to-image models. Our code is publicly available at https://github.com/clear-nus/selective-amnesia.",
        "Unified Concept Editing in Diffusion Models": "Text-to-image models suffer from various safety issues that may limit their suitability for deployment. Previous methods have separately addressed individual issues of bias, copyright, and offensive content in text-to-image models. However, in the real world, all of these issues appear simultaneously in the same model. We present a method that tackles all issues with a single approach. Our method, Unified Concept Editing (UCE), edits the model without training using a closed-form solution, and scales seamlessly to concurrent edits on text-conditional diffusion models. We demonstrate scalable simultaneous debiasing, style erasure, and content moderation by editing text-to-image projections, and we present extensive experiments demonstrating improved efficacy and scalability over prior work. Our code is available at https://unified.baulab.info",
        "Localizing and Editing Knowledge in Text-to-Image Generative Models": "Text-to-Image Diffusion Models such as Stable-Diffusion and Imagen have achieved unprecedented quality of photorealism with state-of-the-art FID scores on MS-COCO and other generation benchmarks. Given a caption, image generation requires fine-grained knowledge about attributes such as object structure, style, and viewpoint amongst others. Where does this information reside in text-to-image generative models? In our paper, we tackle this question and understand how knowledge corresponding to distinct visual attributes is stored in large-scale text-to-image diffusion models. We adapt Causal Mediation Analysis for text-to-image models and trace knowledge about distinct visual attributes to various (causal) components in the (i) UNet and (ii) text-encoder of the diffusion model. In particular, we show that unlike generative large-language models, knowledge about different attributes is not localized in isolated components, but is instead distributed amongst a set of components in the conditional UNet. These sets of components are often distinct for different visual attributes. Remarkably, we find that the CLIP text-encoder in public text-to-image models such as Stable-Diffusion contains only one causal state across different visual attributes, and this is the first self-attention layer corresponding to the last subject token of the attribute in the caption. This is in stark contrast to the causal states in other language models which are often the mid-MLP layers. Based on this observation of only one causal state in the text-encoder, we introduce a fast, data-free model editing method Diff-QuickFix which can effectively edit concepts in text-to-image models. DiffQuickFix can edit (ablate) concepts in under a second with a closed-form update, providing a significant 1000x speedup and comparable editing performance to existing fine-tuning based editing methods.",
        "R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model": r"In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce \textbf{R}obust \textbf{A}dversarial \textbf{C}oncept \textbf{E}rase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges. Our code is publicly available: \url{https://github.com/chkimmmmm/R.A.C.E.}",
        "Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models": "Text-conditioned image generation models have recently achieved astonishing results in image quality and text alignment and are consequently employed in a fast-growing number of applications. Since they are highly data-driven, relying on billion-sized datasets randomly scraped from the internet, they also suffer, as we demonstrate, from degenerated and biased human behavior. In turn, they may even reinforce such biases. To help combat these undesired side effects, we present safe latent diffusion (SLD). Specifically, to measure the inappropriate degeneration due to unfiltered and imbalanced training sets, we establish a novel image generation test bed-inappropriate image prompts (I2P)-containing dedicated, real-world image-to-text prompts covering concepts such as nudity and violence. As our exhaustive empirical evaluation demonstrates, the introduced SLD removes and suppresses inappropriate image parts during the diffusion process, with no additional training required and no adverse effect on overall image quality or text alignment.",
        "SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation": "With evolving data regulations, machine unlearning (MU) has become an important tool for fostering trust and safety in today's AI models. However, existing MU methods focusing on data and/or weight perspectives often suffer limitations in unlearning accuracy, stability, and cross-domain applicability. To address these challenges, we introduce the concept of 'weight saliency' for MU, drawing parallels with input saliency in model explanation. This innovation directs MU's attention toward specific model weights rather than the entire model, improving effectiveness and efficiency. The resultant method that we call saliency unlearning (SalUn) narrows the performance gap with 'exact' unlearning (model retraining from scratch after removing the forgetting data points). To the best of our knowledge, SalUn is the first principled MU approach that can effectively erase the influence of forgetting data, classes, or concepts in both image classification and generation tasks. As highlighted below, For example, SalUn yields a stability advantage in high-variance random data forgetting, e.g., with a 0.2% gap compared to exact unlearning on the CIFAR-10 dataset. Moreover, in preventing conditional diffusion models from generating harmful images, SalUn achieves nearly 100% unlearning accuracy, outperforming current state-of-the-art baselines like Erased Stable Diffusion and Forget-Me-Not. Codes are available at https://github.com/OPTML-Group/Unlearn-Saliency. (WARNING: This paper contains model outputs that may be offensive in nature.)",
        "Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers": "Concept erasure in text-to-image diffusion models aims to disable pre-trained diffusion models from generating images related to a target concept. To perform reliable concept erasure, the properties of robustness and locality are desirable. The former refrains the model from producing images associated with the target concept for any paraphrased or learned prompts, while the latter preserves its ability in generating images with non-target concepts. In this paper, we propose Reliable Concept Erasing via Lightweight Erasers (Receler). It learns a lightweight Eraser to perform concept erasing while satisfying the above desirable properties through the proposed concept-localized regularization and adversarial prompt learning scheme. Experiments with various concepts verify the superiority of Receler over previous methods.",
        "Direct Unlearning Optimization for Robust and Safe Text-to-Image Models": "Recent advancements in text-to-image (T2I) models have greatly benefited from large-scale datasets, but they also pose significant risks due to the potential generation of unsafe content. To mitigate this issue, researchers proposed unlearning techniques that attempt to induce the model to unlearn potentially harmful prompts. However, these methods are easily bypassed by adversarial attacks, making them unreliable for ensuring the safety of generated images. In this paper, we propose Direct Unlearning Optimization (DUO), a novel framework for removing NSFW content from T2I models while preserving their performance on unrelated topics. DUO employs a preference optimization approach using curated paired image data, ensuring that the model learns to remove unsafe visual concepts while retain unrelated features. Furthermore, we introduce an output-preserving regularization term to maintain the model's generative capabilities on safe content. Extensive experiments demonstrate that DUO can robustly defend against various state-of-the-art red teaming methods without significant performance degradation on unrelated topics, as measured by FID and CLIP scores. Our work contributes to the development of safer and more reliable T2I models, paving the way for their responsible deployment in both closed-source and open-source scenarios.",
        "Leveraging Catastrophic Forgetting to Develop Safe Diffusion Models against Malicious Finetuning": "Diffusion models (DMs) have demonstrated remarkable proficiency in producing images based on textual prompts. Numerous methods have been proposed to ensure these models generate safe images. Early methods attempt to incorporate safety filters into models to mitigate the risk of generating harmful images but such external filters do not inherently detoxify the model and can be easily bypassed. Hence, model unlearning and data cleaning are the most essential methods for maintaining the safety of models, given their impact on model parameters. However, malicious fine-tuning can still make models prone to generating harmful or undesirable images even with these methods. Inspired by the phenomenon of catastrophic forgetting, we propose a training policy using contrastive learning to increase the latent space distance between clean and harmful data distribution, thereby protecting models from being fine-tuned to generate harmful images due to forgetting. The experimental results demonstrate that our methods not only maintain clean image generation capabilities before malicious fine-tuning but also effectively prevent DMs from producing harmful images after malicious fine-tuning. Our method can also be combined with other safety methods to maintain their safety against malicious fine-tuning further.",
        "Get What You Want, Not What You Don't: Image Content Suppression for Text-to-Image Diffusion Models": "The success of recent text-to-image diffusion models is largely due to their capacity to be guided by a complex text prompt, which enables users to precisely describe the desired content. However, these models struggle to effectively suppress the generation of undesired content, which is explicitly requested to be omitted from the generated image in the prompt. In this paper, we analyze how to manipulate the text embeddings and remove unwanted content from them. We introduce two contributions, which we refer to as $\textit{soft-weighted regularization}$ and $\textit{inference-time text embedding optimization}$. The first regularizes the text embedding matrix and effectively suppresses the undesired content. The second method aims to further suppress the unwanted content generation of the prompt, and encourages the generation of desired content. We evaluate our method quantitatively and qualitatively on extensive experiments, validating its effectiveness. Furthermore, our method is generalizability to both the pixel-space diffusion models (i.e. DeepFloyd-IF) and the latent-space diffusion models (i.e. Stable Diffusion).",
        "UnlearnCanvas: Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models": "The technological advancements in diffusion models (DMs) have demonstrated unprecedented capabilities in text-to-image generation and are widely used in diverse applications. However, they have also raised significant societal concerns, such as the generation of harmful content and copyright disputes. Machine unlearning (MU) has emerged as a promising solution, capable of removing undesired generative capabilities from DMs. However, existing MU evaluation systems present several key challenges that can result in incomplete and inaccurate assessments. To address these issues, we propose UnlearnCanvas, a comprehensive high-resolution stylized image dataset that facilitates the evaluation of the unlearning of artistic styles and associated objects. This dataset enables the establishment of a standardized, automated evaluation framework with 7 quantitative metrics assessing various aspects of the unlearning performance for DMs. Through extensive experiments, we benchmark 9 state-of-the-art MU methods for DMs, revealing novel insights into their strengths, weaknesses, and underlying mechanisms. Additionally, we explore challenging unlearning scenarios for DMs to evaluate worst-case performance against adversarial prompts, the unlearning of finer-scale concepts, and sequential unlearning. We hope that this study can pave the way for developing more effective, accurate, and robust DM unlearning methods, ensuring safer and more ethical applications of DMs in the future. The dataset, benchmark, and codes are publicly available at https://unlearn-canvas.netlify.app/.",
        "Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation": r"Diffusion-based models have gained significant popularity for text-to-image generation due to their exceptional image-generation capabilities. A risk with these models is the potential generation of inappropriate content, such as biased or harmful images. However, the underlying reasons for generating such undesired content from the perspective of the diffusion model's internal representation remain unclear. Previous work interprets vectors in an interpretable latent space of diffusion models as semantic concepts. However, existing approaches cannot discover directions for arbitrary concepts, such as those related to inappropriate concepts. In this work, we propose a novel self-supervised approach to find interpretable latent directions for a given concept. With the discovered vectors, we further propose a simple approach to mitigate inappropriate generation. Extensive experiments have been conducted to verify the effectiveness of our mitigation approach, namely, for fair generation, safe generation, and responsible text-enhancing generation. Project page: \url{this https URL}.",
        "Defensive Unlearning with Adversarial Training for Robust Concept Erasure in Diffusion Models": "Diffusion models (DMs) have achieved remarkable success in text-to-image generation, but they also pose safety risks, such as the potential generation of harmful content and copyright violations. The techniques of machine unlearning, also known as concept erasing, have been developed to address these risks. However, these techniques remain vulnerable to adversarial prompt attacks, which can prompt DMs post-unlearning to regenerate undesired images containing concepts (such as nudity) meant to be erased. This work aims to enhance the robustness of concept erasing by integrating the principle of adversarial training (AT) into machine unlearning, resulting in the robust unlearning framework referred to as AdvUnlearn. However, achieving this effectively and efficiently is highly nontrivial. First, we find that a straightforward implementation of AT compromises DMs' image generation quality post-unlearning. To address this, we develop a utility-retaining regularization on an additional retain set, optimizing the trade-off between concept erasure robustness and model utility in AdvUnlearn. Moreover, we identify the text encoder as a more suitable module for robustification compared to UNet, ensuring unlearning effectiveness. And the acquired text encoder can serve as a plug-and-play robust unlearner for various DM types. Empirically, we perform extensive experiments to demonstrate the robustness advantage of AdvUnlearn across various DM unlearning scenarios, including the erasure of nudity, objects, and style concepts. In addition to robustness, AdvUnlearn also achieves a balanced tradeoff with model utility. To our knowledge, this is the first work to systematically explore robust DM unlearning through AT, setting it apart from existing methods that overlook robustness in concept erasing. Codes are available at: https://github.com/OPTML-Group/AdvUnlearn",
        "Erasing Undesirable Concepts in Diffusion Models with Adversarial Preservation": r"Diffusion models excel at generating visually striking content from text but can inadvertently produce undesirable or harmful content when trained on unfiltered internet data. A practical solution is to selectively removing target concepts from the model, but this may impact the remaining concepts. Prior approaches have tried to balance this by introducing a loss term to preserve neutral content or a regularization term to minimize changes in the model parameters, yet resolving this trade-off remains challenging. In this work, we propose to identify and preserving concepts most affected by parameter changes, termed as \textit{adversarial concepts}. This approach ensures stable erasure with minimal impact on the other concepts. We demonstrate the effectiveness of our method using the Stable Diffusion model, showing that it outperforms state-of-the-art erasure methods in eliminating unwanted content while maintaining the integrity of other unrelated elements. Our code is available at \url{this https URL}.",
        "Boosting Alignment for Post-Unlearning Text-to-Image Generative Models": r"Large-scale generative models have shown impressive image-generation capabilities, propelled by massive data. However, this often inadvertently leads to the generation of harmful or inappropriate content and raises copyright concerns. Driven by these concerns, machine unlearning has become crucial to effectively purge undesirable knowledge from models. While existing literature has studied various unlearning techniques, these often suffer from either poor unlearning quality or degradation in text-image alignment after unlearning, due to the competitive nature of these objectives. To address these challenges, we propose a framework that seeks an optimal model update at each unlearning iteration, ensuring monotonic improvement on both objectives. We further derive the characterization of such an update. In addition, we design procedures to strategically diversify the unlearning and remaining datasets to boost performance improvement. Our evaluation demonstrates that our method effectively removes target classes from recent diffusion-based generative models and concepts from stable diffusion models while maintaining close alignment with the models' original trained states, thus outperforming state-of-the-art baselines. Our code will be made available at \url{this https URL}.",
        "Unified Gradient-Based Machine Unlearning with Remain Geometry Enhancement": "Machine unlearning (MU) has emerged to enhance the privacy and trustworthiness of deep neural networks. Approximate MU is a practical method for large-scale models. Our investigation into approximate MU starts with identifying the steepest descent direction, minimizing the output Kullback-Leibler divergence to exact MU inside a parameters' neighborhood. This probed direction decomposes into three components: weighted forgetting gradient ascent, fine-tuning retaining gradient descent, and a weight saliency matrix. Such decomposition derived from Euclidean metric encompasses most existing gradient-based MU methods. Nevertheless, adhering to Euclidean space may result in sub-optimal iterative trajectories due to the overlooked geometric structure of the output probability space. We suggest embedding the unlearning update into a manifold rendered by the remaining geometry, incorporating second-order Hessian from the remaining data. It helps prevent effective unlearning from interfering with the retained performance. However, computing the second-order Hessian for large-scale models is intractable. To efficiently leverage the benefits of Hessian modulation, we propose a fast-slow parameter update strategy to implicitly approximate the up-to-date salient unlearning direction. Free from specific modal constraints, our approach is adaptable across computer vision unlearning tasks, including classification and generation. Extensive experiments validate our efficacy and efficiency. Notably, our method successfully performs class-forgetting on ImageNet using DiT and forgets a class on CIFAR-10 using DDPM in just 50 steps, compared to thousands of steps required by previous methods.",
        "GuardT2I: Defending Text-to-Image Models from Adversarial Prompts": r"Recent advancements in Text-to-Image (T2I) models have raised significant safety concerns about their potential misuse for generating inappropriate or Not-Safe-For-Work (NSFW) contents, despite existing countermeasures such as NSFW classifiers or model fine-tuning for inappropriate concept removal. Addressing this challenge, our study unveils GuardT2I, a novel moderation framework that adopts a generative approach to enhance T2I models' robustness against adversarial prompts. Instead of making a binary classification, GuardT2I utilizes a Large Language Model (LLM) to conditionally transform text guidance embeddings within the T2I models into natural language for effective adversarial prompt detection, without compromising the models' inherent performance. Our extensive experiments reveal that GuardT2I outperforms leading commercial solutions like OpenAI-Moderation and Microsoft Azure Moderator by a significant margin across diverse adversarial scenarios. Our framework is available at this https URL.",
        "SAFREE: Training-Free and Adaptive Guard for Safe Text-to-Image And Video Generation": "Recent advances in diffusion models have significantly enhanced their ability to generate high-quality images and videos, but they have also increased the risk of producing unsafe content. Existing unlearning/editing-based methods for safe generation remove harmful concepts from models but face several challenges: (1) They cannot instantly remove harmful concepts without training. (2) Their safe generation capabilities depend on collected training data. (3) They alter model weights, risking degradation in quality for content unrelated to toxic concepts. To address these, we propose SAFREE, a novel, training-free approach for safe T2I and T2V, that does not alter the model's weights. Specifically, we detect a subspace corresponding to a set of toxic concepts in the text embedding space and steer prompt embeddings away from this subspace, thereby filtering out harmful content while preserving intended semantics. To balance the trade-off between filtering toxicity and preserving safe concepts, SAFREE incorporates a novel self-validating filtering mechanism that dynamically adjusts the denoising steps when applying the filtered embeddings. Additionally, we incorporate adaptive re-attention mechanisms within the diffusion latent space to selectively diminish the influence of features related to toxic concepts at the pixel level. In the end, SAFREE ensures coherent safety checking, preserving the fidelity, quality, and safety of the output. SAFREE achieves SOTA performance in suppressing unsafe content in T2I generation compared to training-free baselines and effectively filters targeted concepts while maintaining high-quality images. It also shows competitive results against training-based methods. We extend SAFREE to various T2I backbones and T2V tasks, showcasing its flexibility and generalization. SAFREE provides a robust and adaptable safeguard for ensuring safe visual generation.",
        "Generating Instance-level Prompts for Rehearsal-free Continual Learning": "We introduce Domain-Adaptive Prompt (DAP), a novel method for continual learning using Vision Transformers (ViT). Prompt-based continual learning has recently gained attention due to its rehearsal-free nature. Currently, the prompt pool, which is suggested by prompt-based continual learning, is key to effectively exploiting the frozen pretrained ViT backbone in a sequence of tasks. However, we observe that the use of a prompt pool creates a domain scalability problem between pre-training and continual learning. This problem arises due to the inherent encoding of group-level instructions within the prompt pool. To address this problem, we propose DAP, a pool-free approach that generates a suitable prompt in an instance-level manner at inference time. We optimize an adaptive prompt generator that creates instance-specific fine-grained instructions required for each input, enabling enhanced model plasticity and reduced forgetting. Our experiments on seven datasets with varying degrees of domain similarity to ImageNet demonstrate the superiority of DAP over state-of-the-art prompt-based methods. Code is publicly available at https://github.com/naver-ai/dap-cl.",
    }

    prompt = "Instruct: Retrive papers in similar fields to Queries.\nQueries: \nTitle: {title}\nAbstract: {abstract}"
    normalized_compare_papers = {}
    for title, abstract in compare_papers.items():
        normalized_title = normalize_text(title)
        normalized_abstract = normalize_text(abstract)
        normalized_compare_papers[normalized_title] = normalized_abstract

    compare_papers_list = [prompt.format(title=title, abstract=abstract) for title, abstract in normalized_compare_papers.items()]
    compare_papers_embeddings = get_embeddings(
        compare_papers_list, model, args.batch_size
    )

    normalized_titles = [normalize_text(title) for title in titles]
    normalized_abstracts = [normalize_text(abstract) for abstract in abstracts]
    paper_list_for_embedding = [
        prompt.format(title=title, abstract=abstract)
        for title, abstract in zip(normalized_titles, normalized_abstracts)
    ]

    paper_embeddings = get_embeddings(paper_list_for_embedding, model, args.batch_size)
    
    print("\n유사 논문 검색을 시작합니다.")
    related_papers, num_paper = process_papers(
        titles,
        abstracts,
        paper_embeddings,
        compare_papers_embeddings,
        compare_papers,
        args,
    )
    print(f"\n총 관련 논문 수: {num_paper}")

    if related_papers:
        with alive_bar(1, title="결과 저장 중") as bar:
            output_file, json_output_file = save_results(related_papers, args.file)
            bar()
        print("\n검색 결과가 다음 파일로 저장되었습니다:")
        print(f"- CSV 파일: {output_file}")
        print(f"- JSON 파일: {json_output_file}")


if __name__ == "__main__":
    main()
