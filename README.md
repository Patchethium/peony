# Peony

This is a POC(Proof of Concept) project to demonstrate the possibility of using smaller codebook for generating descrete neural audio codecs.

The model is a simple autoencoder with a bottleneck layer, the middle presentation $`z`$ is shrunk into a lower dimension then use RVQ (Residual Vector Quantization) to get the indices.

We use STE to force $`z`$ to be as close to the codebook at index $`i`$ as possible. The $`i`$ is gained from a larger codebook, which in this case is the neural audio codec extrated from the ground truth audio.

## Training

TODO

<p><a href="https://commons.wikimedia.org/wiki/File:PaeoniaSuffruticosa7.jpg#/media/File:PaeoniaSuffruticosa7.jpg"><img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/PaeoniaSuffruticosa7.jpg" alt="PaeoniaSuffruticosa7.jpg"></a><br>By Taken by <a href="//commons.wikimedia.org/wiki/User:Fanghong" title="User:Fanghong">Fanghong</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by/2.5" title="Creative Commons Attribution 2.5">CC BY 2.5</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=733608">Link</a></p>