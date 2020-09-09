# Subspace-Learning-for-Person-Re-identification
### Authors: Neelabhro Roy, Dr. AV Subramanyam (Advisor, Associate Professor, IIIT-Delhi)
### Abstract:
Feature transformation has shown a promising performance in person re-identification by bridging the gap in inconsistent distribution of features across different non-overlapping cameras. However, this is either performed locally or globally. In this work, we propose a global and local feature transformation method. The global feature transformation matrix projects the data from different cameras to a common space. We further hypothesize that a latent basis matrix can be learnt in this space which represents the shared structure between different cameras using matrix factorization. The factorization also yields semantic representations of identities. In order to better generalize to out-of-sample instances, we also learn local semantic projection matrices. Further, to incorporate the mapping between cameras for a given identity, we learn an association matrix. We frame a joint optimization problem and show the analytical proof for obtaining the solution. Extensive experimental results reveal that the proposed algorithm outperforms many popular algorithms. 
Experiments were performed in the following datasets. The work can be found here: https://drive.google.com/file/d/1ncUd2X23REuf2k5pfranxIKHUKLc58EH/view?usp=sharing

### Dataset: VIPeR
![](index.jpeg)


### Dataset: VIPeR (Left) + PRID450S (Right)
![](index2.jpeg)


### Dataset: CUHK01
![](index3.jpeg)

### Conclusion
In this paper, we propose a global and local feature transformation algorithm for cross-view person re-identification. The global transformation helps in reducing the inconsistency in paired feature descriptors across cameras. The common intrinsic structure is explored via a shared semantic basis matrix. The local projection matrices help in generalizing to out-of-sample instances. Further, the association matrix learns the relationship between semantic representations.
Experimental results reveal superior performance obtained by our algorithm.
