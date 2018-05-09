import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import statsmodels.api as sm
from astropy.table import Table
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas

# Final Code For Word Recall Full Model

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

files_ltpFR2 = glob.glob(    '/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP[0-9][0-9][0-9].json')
#files_ltpFR2 = glob.glob('/Users/adaaka/rhino_mount/data/eeg/scalp/ltp/ltpFR2/behavioral/data/beh_data_LTP22[0-9].json')

"""p_rec_word = p_rec_word.get_ltpFR2_precall(files_ltpFR2)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
valence_norm = conc_freq_len.valences_norm
arousal_norm = conc_freq_len.arousals_norm
m_list = m_list.all_parts_list_correlations(files_ltpFR2)
m_pool = m_pool.w2v_filtered_corr

all_probs = np.array(p_rec_word)
all_concreteness = np.array(concreteness_norm)
all_wordfreq = np.array(word_freq_norm)
all_wordlen = np.array(word_length_norm)
all_valence = np.array(valence_norm)
all_arousal = np.array(arousal_norm)
all_m_list = np.array(m_list)
all_means_mpool = np.array(m_pool)

all_probs_part = []
all_concreteness_part = []
all_wordfreq_part = []
all_wordlen_part = []
all_valence_part = []
all_arousal_part = []
all_m_list_part = []
all_means_mpool_part = []

params = []
rsquareds = []
predict = []
pvalues = []
residuals = []
f_pvalues = []
fdr_pvalues = []


def calculate_params():
    # for each participant (= each list in list of lists)
    for i in range(len(all_probs)):
        all_probs_part = all_probs[i]
        print("LOOK HERE:", all_probs_part)
        all_probs_part_norm = stats.mstats.zscore(all_probs_part, axis=0, ddof=1)
        all_concreteness_part = all_concreteness
        all_wordfreq_part = all_wordfreq
        all_wordlen_part = all_wordlen
        all_valence_part = all_valence
        all_arousal_part = all_arousal
        all_m_list_part = all_m_list[i]
        all_m_list_part_norm = stats.mstats.zscore(all_m_list_part, axis=0, ddof=1)
        all_means_mpool_part = all_means_mpool
        all_means_mpool_part_norm = stats.mstats.zscore(all_means_mpool_part, axis=0, ddof=1)

        # print(all_imageability)
        mask = np.logical_not(np.isnan(all_concreteness_part))
        # print("Mask", mask)
        # print(np.shape(np.array(mask)))
        all_probs_part_norm = np.array(all_probs_part_norm)[mask]
        all_concreteness_part = np.array(all_concreteness_part)[mask]
        all_wordfreq_part = np.array(all_wordfreq_part)[mask]
        all_wordlen_part = np.array(all_wordlen_part)[mask]
        all_valence_part_norm = np.array(all_valence_part)[mask]
        all_arousal_part_norm = np.array(all_arousal_part)[mask]
        all_m_list_part_norm = np.array(all_m_list_part_norm)[mask]
        all_means_mpool_part_norm = np.array(all_means_mpool_part_norm)[mask]

        # print(len(all_probs_part_norm))
        # print(len(all_concreteness_part))
        # print(len(all_wordfreq_part))
        # print(len(all_wordlen_part))
        # print(len(all_valence_part_norm))
        # print(len(all_arousal_part_norm))
        # print(len(all_imag_part_norm))
        # print(len(all_m_list_part_norm))
        # print(len(all_means_mpool_part_norm))
        #
        # print((all_probs_part_norm))
        # print((all_concreteness_part))
        # print((all_wordfreq_part))
        # print((all_wordlen_part))
        # print((all_valence_part_norm))
        # print((all_arousal_part_norm))
        # print((all_imag_part_norm))
        # print((all_m_list_part_norm))
        # print((all_means_mpool_part_norm))

        sm.OLS

        x_values = np.column_stack((all_concreteness_part, all_wordfreq_part, all_wordlen_part, all_valence_part_norm,
                                    all_arousal_part_norm, all_m_list_part_norm, all_means_mpool_part_norm))

        x_values = sm.add_constant(x_values)

        y_value = all_probs_part_norm

        model = sm.OLS(y_value, x_values)
        results = model.fit()

        print(results.summary())
        params.append(results.params)
        rsquareds.append(results.rsquared)
        predict.append(results.predict(x_values))
        pvalues.append(results.pvalues)
        fdr_p = (multipletests([results.pvalues[0], results.pvalues[1], results.pvalues[2], results.pvalues[3],
                                results.pvalues[4], results.pvalues[5], results.pvalues[6], results.pvalues[7]],
                               alpha=0.05,
                               method='fdr_bh', is_sorted=False, returnsorted=False))

        fdr_pvalues.append(fdr_p[0])
        f_pvalues.append(results.f_pvalue)
        residuals.append(results.resid)

    return params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues


params, rsquareds, predict, pvalues, residuals, f_pvalues, fdr_pvalues = calculate_params()

rsquareds = np.array(rsquareds)
params = np.array(params)
predict = np.array(predict)
pvalues = np.array(pvalues)
fdr_pvalues = np.array(fdr_pvalues)
print(fdr_pvalues.shape)

residuals = np.array(residuals)

rsquareds_sig = []
rsquareds_notsig = []

for i in range(0, len(all_probs)):
    if f_pvalues[i] < .05:
        rsquareds_sig.append(rsquareds[i])
    else:
        rsquareds_notsig.append(rsquareds[i])
print("Significant R-Squareds", rsquareds_sig)
print("Not Significant R-Squareds", rsquareds_notsig)

x_rsq_sig = np.full(len(rsquareds_sig), 1)
x_rsq_nonsig = np.full(len(rsquareds_notsig), 1)

print("R sig", len(rsquareds_sig))
print("R not sig", len(rsquareds_notsig))

sig = np.array(rsquareds_sig)
no = np.array(rsquareds_notsig)

intercepts = np.array(params[:,1])
beta_concreteness = (np.array(params[:, 1]))
beta_wordfreq = (np.array(params[:, 2]))
beta_wordlen = (np.array(params[:, 3]))
beta_valence = (np.array(params[:, 4]))
beta_arousal = (np.array(params[:, 5]))
beta_mlist = (np.array(params[:, 6]))
beta_mpool = (np.array(params[:, 7]))

mean_intercept = np.mean(intercepts)
mean_beta_conc = np.mean(beta_concreteness)
mean_beta_wordfreq = np.mean(beta_wordfreq)
mean_beta_wordlen = np.mean(beta_wordlen)
mean_beta_valence = np.mean(beta_valence)
mean_beta_arousal = np.mean(beta_arousal)
mean_beta_mlist = np.mean(beta_mlist)
mean_beta_mpool = np.mean(beta_mpool)

print("Mean int", mean_intercept)
print("Mean conc", mean_beta_conc)
print("Mean wordfreq", mean_beta_wordfreq)
print("Mean wordlen", mean_beta_wordlen)
print("Mean valence", mean_beta_valence)
print("Mean arousal", mean_beta_arousal)
print("Mean mlist", mean_beta_mlist)
print("Mean mpool", mean_beta_mpool)

"""

mean_intercept = -0.00504647026906
mean_beta_conc = -0.00504647026906
mean_beta_wordfreq = 0.0611066377604
mean_beta_wordlen = -0.00675036022394
mean_beta_valence =0.107557510562
mean_beta_arousal =0.129507157582
mean_beta_mlist =0.0105579379656
mean_beta_mpool =0.131500847487

from w2vec.Final import Concreteness_Freq_WordLength as conc_freq_len, ltpFR2_word_to_allwords as m_pool, \
    P_rec_ltpFR2_words as p_rec_word, semanticsim_each_word as m_list

all_probs = [0.5568855552	,
0.614310097	,
0.4775529148	,
0.5496990658	,
0.5678651635	,
0.5771982958	,
0.4869969457	,
0.5711687028	,
0.5236704995	,
0.5863950773	,
0.577157444	,
0.6500938949	,
0.4727811714	,
0.4993262666	,
0.4892427237	,
0.4808210564	,
0.6101329501	,
0.448189903	,
0.5538312972	,
0.4390495868	,
0.4565666547	,
0.459708498	,
0.5022502695	,
0.5370660301	,
0.5285213798	,
0.4535573123	,
0.4741778314	,
0.5499387223	,
0.4906800216	,
0.4752066116	,
0.5035483291	,
0.5394134028	,
0.5671936759	,
0.5389139846	,
0.4578242903	,
0.4181863097	,
0.5876976285	,
0.5054796982	,
0.5274658642	,
0.4435272402	,
0.463865433	,
0.5393011139	,
0.4433390226	,
0.4712818473	,
0.5044915559	,
0.4810905498	,
0.5428505125	,
0.6864220266	,
0.4299991017	,
0.3751208442	,
0.4221452954	,
0.5149568811	,
0.5666322314	,
0.4858740568	,
0.4893453878	,
0.4539679688	,
0.6556997844	,
0.4336821775	,
0.4541636723	,
0.5171502575	,
0.5048059648	,
0.5113007546	,
0.5508444125	,
0.4822134387	,
0.5234683794	,
0.5423104563	,
0.5335022789	,
0.5336866691	,
0.4870194035	,
0.490432986	,
0.4671419048	,
0.4669507041	,
0.5123581952	,
0.4547924901	,
0.5307671577	,
0.5238501617	,
0.4791591807	,
0.4720625225	,
0.5322942867	,
0.4337720086	,
0.5410303629	,
0.5289769519	,
0.5123877111	,
0.5624775422	,
0.5013014818	,
0.5733695652	,
0.5443990298	,
0.5114663005	,
0.587136184	,
0.5220310816	,
0.5431863097	,
0.433839382	,
0.5124640676	,
0.5257815307	,
0.4104223346	,
0.4938016529	,
0.3640630614	,
0.4824829321	,
0.4718379447	,
0.5187362045	,
0.4757680561	,
0.4574200503	,
0.6052596119	,
0.4886491967	,
0.4897367948	,
0.587742544	,
0.5715729429	,
0.4674362199	,
0.3867454186	,
0.4665603665	,
0.4111121092	,
0.5554931728	,
0.5364399928	,
0.5232662594	,
0.4314267748	,
0.4324919152	,
0.5285887531	,
0.5191447222	,
0.4982258354	,
0.417728171	,
0.4203711308	,
0.5793433345	,
0.5666097736	,
0.5160573123	,
0.4737842855	,
0.4546823401	,
0.4720261622	,
0.5593783687	,
0.5569304707	,
0.4290783327	,
0.4415199425	,
0.4694574201	,
0.546543213	,
0.4203647143	,
0.4795719761	,
0.4720774943	,
0.4812252964	,
0.4808884298	,
0.5180111391	,
0.5251302551	,
0.432110133	,
0.4800125764	,
0.6462226015	,
0.5206836148	,
0.5646559468	,
0.4749820338	,
0.5166422839	,
0.5050979159	,
0.5713012037	,
0.5291726554	,
0.4602497305	,
0.5461282788	,
0.4866098164	,
0.5437926698	,
0.4421263026	,
0.5430740208	,
0.4902533238	,
0.4423861711	,
0.4400238266	,
0.5238565782	,
0.4255300036	,
0.5242768595	,
0.4707375135	,
0.5085339562	,
0.5929752066	,
0.5360671937	,
0.4443047072	,
0.5939633489	,
0.4798913043	,
0.439431369	,
0.5596703198	,
0.5747619475	,
0.4289885016	,
0.4237333812	,
0.5400197628	,
0.5700008983	,
0.5254260562	,
0.4079295211	,
0.431593604	,
0.4855459593	,
0.4901634926	,
0.4601053591	,
0.4238905857	,
0.5138789077	,
0.4347954417	,
0.5425596308	,
0.5013474668	,
0.497349982	,
0.5369879626	,
0.5464309241	,
0.5014148401	,
0.4395212001	,
0.5887114368	,
0.4582734459	,
0.4620238951	,
0.5954904779	,
0.573998383	,
0.5317103845	,
0.4731404959	,
0.4510644987	,
0.550080848	,
0.4264283148	,
0.5526859504	,
0.4944304707	,
0.4993262666	,
0.5659959277	,
0.5100910289	,
0.4376572045	,
0.6685231764	,
0.5162594323	,
0.4567014014	,
0.4510644987	,
0.4995059289	,
0.5084975959	,
0.5118234861	,
0.5139366562	,
0.4870867769	,
0.4635285663	,
0.5650152713	,
0.461935775	,
0.5052186532	,
0.4431593604	,
0.5121721164	,
0.431795724	,
0.5291501976	,
0.4777443406	,
0.4009857896	,
0.4975970176	,
0.5161920589	,
0.4761947539	,
0.4947512961	,
0.4676447564	,
0.4476284585	,
0.4989444844	,
0.5369655049	,
0.5484788597	,
0.5155899766	,
0.4440352138	,
0.4339580874	,
0.4551951542	,
0.5732797341	,
0.5218738771	,
0.5357977003	,
0.5248393734	,
0.651365433	,
0.5094140872	,
0.479136723	,
0.4640450952	,
0.553539346	,
0.6129241312	,
0.5636111254	,
0.5333273446	,
0.3982363157	,
0.6279397233	,
0.5310141933	,
0.4963169242	,
0.521581926	,
0.4527488322	,
0.5364297264	,
0.4876931369	,
0.4997754222	,
0.5279599353	,
0.4439678405	,
0.5523482282	,
0.4965190442	,
0.5605910888	,
0.4536246856	,
0.4948796263	,
0.5592211642	,
0.3635080335	,
0.5388829714	,
0.4929482573	,
0.6069888609	,
0.5337765002	,
0.5650377291	,
0.3866780453	,
0.5029751211	,
0.4375021388	,
0.5076581028	,
0.4678778896	,
0.5696864894	,
0.5727182896	,
0.5389193317	,
0.5212899748	,
0.5681518745	,
0.57797341	,
0.5358650737	,
0.5024703557	,
0.5358426159	,
0.4746334035	,
0.4442822494	,
0.630704276	,
0.6225633309	,
0.4941160618	,
0.4529060367	,
0.4881647503	,
0.4620987543	,
0.4815300806	,
0.5737620416	,
0.4861467584	,
0.4221003799	,
0.5223679483	,
0.415311714	,
0.4594883048	,
0.4385191554	,
0.5437766285	,
0.5126522851	,
0.4467975207	,
0.4031822673	,
0.5345400647	,
0.583161157	,
0.4517831477	,
0.4338169242	,
0.4831117499	,
0.5271289975	,
0.5265900108	,
0.4906800216	,
0.4968377308	,
0.5076805605	,
0.4908147682	,
0.4809108875	,
0.40554258	,
0.4520077255	,
0.546172125	,
0.5297565577	,
0.4131378009	,
0.4874236436	,
0.4612827884	,
0.5344277758	,
0.5135955119	,
0.4627286416	,
0.4325368308	,
0.5871137262	,
0.5109593963	,
0.4883732868	,
0.5354293927	,
0.5303404599	,
0.5797475746	,
0.5230438205	,
0.4692617165	,
0.4565891125	,
0.5258938196	,
0.6528251886	,
0.5680021559	,
0.4620238951	,
0.4567238591	,
0.4457954759	,
0.4337046353	,
0.5449380165	,
0.5578736974	,
0.5168882501	,
0.4647255659	,
0.5976913403	,
0.443772137	,
0.5194933525	,
0.5157878189	,
0.43810636	,
0.4593963349	,
0.4702883579	,
0.4621810995	,
0.4733875314	,
0.5524838304	,
0.4776994251	,
0.5039525692	,
0.4606422069	,
0.4190236641	,
0.5471956144	,
0.4770256917	,
0.5058390226	,
0.4422161337	,
0.3819212737	,
0.4589696371	,
0.529621811	,
0.4237333812	,
0.5129333169	,
0.408282429	,
0.5470939633	,
0.4789346029	,
0.5331925979	,
0.4544331656	,
0.5349710402	,
0.546361412	,
0.4266753503	,
0.498472871	,
0.5499236436	,
0.4812477542	,
0.5155407833	,
0.5084665828	,
0.4463729617	,
0.4603620194	,
0.5244789795	,
0.6081342077	,
0.4158282429	,
0.5181009702	,
0.4634836507	,
0.4938102082	,
0.4255749192	,
0.4545806384	,
0.5697987783	,
0.4409809558	,
0.6036651096	,
0.6749910169	,
0.4666277398	,
0.5005839023	,
0.4706701401	,
0.5164027343	,
0.5900556953	,
0.5503503414	,
0.4360220642	,
0.6868711822	,
0.5326086957	,
0.3805470715	,
0.4771604384	,
0.48671355	,
0.4355461732	,
0.4754761049	,
0.4392784422	,
0.4198032699	,
0.35932447	,
0.4916906216	,
0.4188375853	,
0.5606360043	,
0.5308794466	,
0.5210204815	,
0.4707599713	,
0.4352317643	,
0.4972944286	,
0.5261312304	,
0.4659764642	,
0.5185501258	,
0.4818188235	,
0.459649787	,
0.5077030183	,
0.5440814127	,
0.5103754941	,
0.5143729788	,
0.555785124	,
0.5259836507	,
0.5658911247	,
0.4579141215	,
0.4588680424	,
0.4753862738	,
0.4771222602	,
0.440162851	,
0.4580713259	,
0.5388968739	,
0.5025601868	,
0.5105369762	,
0.480147323	,
0.5176293568	,
0.5382680561	,
0.4706316411	,
0.4220041322	,
0.579904779	,
0.5355731225	,
0.4976216142	,
0.538725767	,
0.526365433	,
0.4410932447	,
0.4252155947	,
0.5981404959	,
0.5348993891	,
0.5289031621	,
0.6729249012	,
0.4095400647	,
0.5386049227	,
0.5584575997	,
0.4580798813	,
0.532878189	,
0.4685141933	,
0.523787066	,
0.4438555516	,
0.4476284585	,
0.4954474873	,
0.5300709666	,
0.4151387891	,
0.4699782266	,
0.4969008264	,
0.5132725476	,
0.5730326985	,
0.4869520302	,
0.3964696371	,
0.4654150198	,
0.4741735537	,
0.587885846	,
0.4434513115	,
0.5676428315	,
0.3991421128	,
0.3887751741	,
0.4991016888	,
0.4390945023	,
0.366699177	,
0.4941160618	,
0.4869295724	,
0.4892470014	,
0.5052743699	,
0.6172745239	,
0.4828957275	,
0.4440191725	,
0.5486660079	,
0.4894897593	,
0.5098814229	,
0.4289435861	,
0.3505502156	,
0.505704276	,
0.5037344079	,
0.5336866691	,
0.3718783687	,
0.5205488681	,
0.5011111254	,
0.3888382698	,
0.4713101868	,
0.5167759612	,
0.6180894205	,
0.5003368667	,
0.475363816	,
0.5648131513	,
0.4454693034	,
0.5414795185	,
0.4865702479	,
0.4205037387	,
0.5456117499	,
0.5546397772	,
0.5003047841	,
0.4895346748	,
0.5351239669	,
0.4142561983	,
0.4750376435	,
0.4619832572	,
0.5029419691	,
0.5474372465	,
0.5033754042	,
0.4439229249	,
0.506512756	,
0.5139120596	,
0.4877508855	,
0.5188645347	,
0.5362019404	,
0.4807985986	,
0.4873423678	,
0.4624505929	,
0.5417265541	,
0.4807761409	,
0.4928404599	,
0.6080443766	,
0.5538762127	,
0.5030318002	,
0.4080039526	,
0.5226958318	,
0.5339978697	,
0.4975071865	,
0.5229967661	,
0.4505704276	,
0.4587899748	,
0.5359773626	,
0.615837226	,
0.4857393101	,
0.4121023904	,
0.4652578153	,
0.4834935322	,
0.3659495149	,
0.54722871	,
0.6776560067	,
0.5353752802	,
0.4427326626	,
0.6472332016	,
0.4765540783	,
0.4803269853	,
0.4620238951	,
0.5592660798	,
0.490942029	,
0.5563465685	,
0.4417669781	,
0.4695921667	,
0.568810636	,
0.4732592013	]

all_probs = np.array(all_probs)
concreteness_norm = conc_freq_len.concreteness_norm
word_freq_norm = conc_freq_len.word_freq_norm
word_length_norm = conc_freq_len.word_length_norm
valence_norm = conc_freq_len.valences_norm
arousal_norm = conc_freq_len.arousals_norm

m_list = [0.1267616328	,
0.1238217736	,
0.1218367649	,
0.1282760883	,
0.1248888298	,
0.1254673955	,
0.1270949632	,
0.1307618935	,
0.1295845315	,
0.1252458265	,
0.1232208166	,
0.120539586	,
0.1227712167	,
0.1232559562	,
0.1203780589	,
0.1194468418	,
0.1289623621	,
0.1213247299	,
0.1288432552	,
0.1231402416	,
0.1260886958	,
0.1274643051	,
0.1251288539	,
0.1228150577	,
0.1253928631	,
0.1265681233	,
0.1251674796	,
0.1270360215	,
0.1313456195	,
0.126355284	,
0.1289146747	,
0.1304214718	,
0.1255592973	,
0.1299047663	,
0.1279055728	,
0.1243477155	,
0.1272502286	,
0.1250473985	,
0.1299880696	,
0.127339987	,
0.119478697	,
0.1287129517	,
0.1210135709	,
0.1224870919	,
0.1257157503	,
0.1262059814	,
0.1245531054	,
0.12697577	,
0.1243875257	,
0.1214512819	,
0.1217354114	,
0.1283701213	,
0.1262492428	,
0.1254251595	,
0.125702111	,
0.124641533	,
0.1263486492	,
0.122627764	,
0.1280953685	,
0.1271316513	,
0.1254656203	,
0.1274616282	,
0.1301955026	,
0.1218391486	,
0.1271637003	,
0.1271628041	,
0.1310992723	,
0.1270807291	,
0.128852877	,
0.1260171982	,
0.1259020032	,
0.1262383351	,
0.1252151772	,
0.1197663247	,
0.1266805027	,
0.1286483337	,
0.1268725501	,
0.1258476803	,
0.1268442204	,
0.1256276171	,
0.1284534014	,
0.1279599029	,
0.1252270757	,
0.1249354935	,
0.1265531397	,
0.1257750218	,
0.1213582935	,
0.1290157456	,
0.1289193088	,
0.1287898456	,
0.1258284457	,
0.1300102893	,
0.1289163422	,
0.1280493786	,
0.1241489341	,
0.1253132823	,
0.1230323461	,
0.1274160649	,
0.1224545864	,
0.1315541205	,
0.1255995512	,
0.1259638799	,
0.1261488183	,
0.1262013338	,
0.1256076789	,
0.1206716352	,
0.1243868098	,
0.1261132567	,
0.1256942567	,
0.1226604633	,
0.1208621768	,
0.1233414893	,
0.128645118	,
0.1284447213	,
0.1277255964	,
0.1286247305	,
0.1273209479	,
0.1286962735	,
0.1212392365	,
0.1220095955	,
0.1195291073	,
0.1275251863	,
0.126126422	,
0.1311753149	,
0.124540831	,
0.1286372664	,
0.1310794362	,
0.1272178491	,
0.1303126198	,
0.1249432241	,
0.1209724886	,
0.1200846677	,
0.1237446752	,
0.1267066609	,
0.1273971808	,
0.1244414953	,
0.130211419	,
0.1207642641	,
0.1295124917	,
0.1268890993	,
0.1273789595	,
0.126410818	,
0.1281967073	,
0.1247761897	,
0.1259767575	,
0.1249706285	,
0.1277949678	,
0.1251829941	,
0.127848583	,
0.1283155051	,
0.1220124831	,
0.1249389239	,
0.1279579865	,
0.1280664367	,
0.1184071107	,
0.1287835709	,
0.1287749405	,
0.1234197793	,
0.1286950253	,
0.1232847306	,
0.1200830372	,
0.1267791271	,
0.1266015827	,
0.1288572009	,
0.1245385302	,
0.1256876542	,
0.121834937	,
0.1197815968	,
0.1205968188	,
0.1240926997	,
0.1272807025	,
0.1260199141	,
0.1228000318	,
0.1246722032	,
0.126028739	,
0.1292229501	,
0.1279225038	,
0.124076031	,
0.1275524032	,
0.1284662631	,
0.1194496406	,
0.1304078729	,
0.128138589	,
0.1324321125	,
0.1277912896	,
0.1250418383	,
0.1223849117	,
0.1305847164	,
0.1251449099	,
0.1288699298	,
0.1243213191	,
0.1228836183	,
0.1184944463	,
0.13080754	,
0.1289830118	,
0.1257916863	,
0.1273872198	,
0.1294353254	,
0.1282491881	,
0.1276712752	,
0.127656301	,
0.1251713645	,
0.1300708258	,
0.1307463872	,
0.124367849	,
0.126748507	,
0.1247685055	,
0.1257191421	,
0.1296262928	,
0.1285876184	,
0.1194059954	,
0.1280796077	,
0.1271212023	,
0.1282821154	,
0.1256752484	,
0.1246998014	,
0.1262690653	,
0.1301082683	,
0.1257474184	,
0.1244424636	,
0.1241501676	,
0.128220961	,
0.1250926527	,
0.1205627533	,
0.1258173116	,
0.1300318422	,
0.1272570026	,
0.1245076902	,
0.1249935896	,
0.1292041024	,
0.1258095747	,
0.1239868501	,
0.1286467757	,
0.124091208	,
0.1225976437	,
0.1250411303	,
0.1298314032	,
0.1281165797	,
0.1248953631	,
0.1286533662	,
0.1296273351	,
0.125021344	,
0.1256206002	,
0.1233981173	,
0.1296045622	,
0.1223001926	,
0.1241422994	,
0.1248962274	,
0.1264402934	,
0.1301389396	,
0.1238355185	,
0.124277672	,
0.1263017333	,
0.1207129443	,
0.1288921949	,
0.1308209059	,
0.1303638419	,
0.1231127236	,
0.1243822103	,
0.1265823889	,
0.1283760243	,
0.1234389742	,
0.1251065804	,
0.1250496886	,
0.1258538155	,
0.1323644373	,
0.1296397707	,
0.1281244108	,
0.1302630585	,
0.1230178316	,
0.1278288185	,
0.1235463748	,
0.1286796706	,
0.1279292026	,
0.1268744688	,
0.1278266161	,
0.1264765319	,
0.1224319061	,
0.1268242726	,
0.1200043441	,
0.125233779	,
0.1286026243	,
0.1296710913	,
0.1235567227	,
0.1277976093	,
0.1254382808	,
0.128476935	,
0.1282432308	,
0.1320128153	,
0.1268007553	,
0.1270526041	,
0.1204350777	,
0.127461336	,
0.1273241297	,
0.1242405848	,
0.1239900465	,
0.1251069995	,
0.1253184241	,
0.1269611998	,
0.127668985	,
0.126688456	,
0.1219026611	,
0.1248737696	,
0.1191133871	,
0.1236743576	,
0.1211082067	,
0.1250616701	,
0.1302752264	,
0.1213541516	,
0.1288263296	,
0.1289108027	,
0.1223840499	,
0.121125146	,
0.1275339768	,
0.1279593354	,
0.1261596921	,
0.1278542534	,
0.1214712232	,
0.129390861	,
0.1239211806	,
0.1268472705	,
0.125490362	,
0.1334320394	,
0.1300148957	,
0.1260719423	,
0.127428869	,
0.1271763792	,
0.1212843017	,
0.1327021537	,
0.1248180105	,
0.1226030765	,
0.1206439212	,
0.1259246121	,
0.1230127789	,
0.1273091594	,
0.1295236442	,
0.1303716135	,
0.1293760667	,
0.1293159891	,
0.1248923313	,
0.1236242239	,
0.1221203492	,
0.1273758459	,
0.1291649168	,
0.1224009989	,
0.1214075284	,
0.1256138974	,
0.1277678971	,
0.1298276678	,
0.1270652066	,
0.1288243685	,
0.1249896348	,
0.1242618347	,
0.1273341653	,
0.1311469957	,
0.1197785791	,
0.1175840474	,
0.1281952493	,
0.1311663947	,
0.1226888472	,
0.1202023501	,
0.1306570753	,
0.1263024316	,
0.1299207251	,
0.1277700428	,
0.12514243	,
0.1302592394	,
0.1267056274	,
0.1258551655	,
0.1265131303	,
0.1219494781	,
0.1271817275	,
0.1267747621	,
0.1250386291	,
0.1294966998	,
0.1253880726	,
0.1213094939	,
0.1303755903	,
0.1273900707	,
0.1247811723	,
0.1313479473	,
0.1281925231	,
0.1276212064	,
0.1272418416	,
0.1225744024	,
0.1228059711	,
0.1271063733	,
0.1198430415	,
0.1282587348	,
0.1232153816	,
0.1274355651	,
0.1228544477	,
0.1293985702	,
0.1279902247	,
0.1250059824	,
0.1296176399	,
0.1213804985	,
0.1268159555	,
0.1244879543	,
0.1264366827	,
0.1263121298	,
0.1277291526	,
0.1218612841	,
0.1304877808	,
0.129849912	,
0.1226214336	,
0.1322191617	,
0.1307906626	,
0.1181342052	,
0.1277455604	,
0.1326368508	,
0.1249580569	,
0.1319022799	,
0.1224787313	,
0.1257030921	,
0.1278675949	,
0.1244350638	,
0.1211733824	,
0.1191124956	,
0.1230982887	,
0.1189919518	,
0.1275778041	,
0.1261692522	,
0.127596619	,
0.1288989163	,
0.1286601884	,
0.1219022974	,
0.1284331589	,
0.1305875641	,
0.126297688	,
0.1252641758	,
0.1253284306	,
0.1314659678	,
0.1296288235	,
0.1262681063	,
0.129497573	,
0.1308480513	,
0.1283917461	,
0.1271504564	,
0.1289832414	,
0.1302833072	,
0.127578303	,
0.1271109492	,
0.1247844721	,
0.1290147236	,
0.1264832085	,
0.128755647	,
0.1232141048	,
0.1244678736	,
0.1261986319	,
0.1248091424	,
0.1236713176	,
0.1217111699	,
0.1229096759	,
0.1281592835	,
0.1310679289	,
0.1252894804	,
0.1247006178	,
0.1264584767	,
0.1285684365	,
0.1257369256	,
0.1256552055	,
0.1278060293	,
0.1268298204	,
0.1246252414	,
0.1282136473	,
0.1275378495	,
0.1219578626	,
0.1265560189	,
0.1272453641	,
0.1262270254	,
0.1236185398	,
0.1274938304	,
0.1306839781	,
0.1319639006	,
0.1270382103	,
0.1249929953	,
0.1202107333	,
0.1277178566	,
0.1288175682	,
0.129706578	,
0.1286306452	,
0.1271521161	,
0.1327396804	,
0.1265369409	,
0.1196153999	,
0.1231781264	,
0.1238431545	,
0.1218504844	,
0.120495438	,
0.1277101862	,
0.1267221582	,
0.1278694519	,
0.1308136156	,
0.1306268055	,
0.1216778483	,
0.1240295038	,
0.1239160973	,
0.12861672	,
0.1200555309	,
0.1247021724	,
0.1255145622	,
0.1272828499	,
0.1201006601	,
0.123655845	,
0.1282158474	,
0.1244745239	,
0.1210900369	,
0.1309145608	,
0.1240127324	,
0.1243347221	,
0.1283302861	,
0.1241439054	,
0.1244436829	,
0.1266911991	,
0.1263604505	,
0.127842426	,
0.1248257525	,
0.1251138626	,
0.1273509752	,
0.1300340846	,
0.1275467371	,
0.1275860274	,
0.1246949296	,
0.1285585616	,
0.1247002374	,
0.124568526	,
0.1268967638	,
0.1255081906	,
0.1236894426	,
0.1290713696	,
0.1296089633	,
0.1255732984	,
0.1255951113	,
0.1317964401	,
0.1285569598	,
0.1323156957	,
0.129008049	,
0.1279774139	,
0.1295213904	,
0.1257256776	,
0.1248268322	,
0.1239593569	,
0.1225289598	,
0.1258297268	,
0.1291593289	,
0.123305792	,
0.1257548714	,
0.1294797853	,
0.1232248059	,
0.125639556	,
0.1263649722	,
0.1248972256	,
0.1275642482	,
0.1266393953	,
0.1297074532	,
0.1275532748	,
0.1279686711	,
0.1298692699	,
0.1294032079	,
0.1253623586	,
0.1290098015	,
0.1249231697	,
0.1248782292	,
0.1230015852	,
0.1307738412	,
0.124557105	,
0.120616444	,
0.1272351439	,
0.1260472908	,
0.1252230303	,
0.1224389614	,
0.1254566001	,
0.1284436134	,
0.1273562468	,
0.1279951408	]
m_pool = m_pool.w2v_filtered_corr


all_concreteness_ = np.array(concreteness_norm)
all_wordfreq_ = np.array(word_freq_norm)
all_wordlen_ = np.array(word_length_norm)
all_valence_ = np.array(valence_norm)
all_arousal_ = np.array(arousal_norm)
all_m_list_ = np.array(m_list)
all_means_mpool_ = np.array(m_pool)

mask = np.logical_not(np.isnan(all_concreteness_))
# print("Mask", mask)
# print(np.shape(np.array(mask)))
all_probs_ = np.array(all_probs)[mask]
all_concreteness_ = np.array(all_concreteness_)[mask]
all_wordfreq_ = np.array(all_wordfreq_)[mask]
all_wordlen_ = np.array(all_wordlen_)[mask]
all_valence_ = np.array(all_valence_)[mask]
all_arousal_ = np.array(all_arousal_)[mask]
all_m_list_ = np.array(all_m_list_)[mask]
all_means_mpool_ = np.array(all_means_mpool_)[mask]

print(len(all_probs_))
print(len(all_concreteness_))
print(len(all_wordfreq_))
print(len(all_wordlen_))
print(len(all_valence_))
print(len(all_arousal_))
print(len(all_m_list_))
print(len(all_means_mpool_))

pred = []
for i in range(568):
    pred.append((mean_beta_conc * all_concreteness_[i]) + (mean_beta_wordfreq * all_wordfreq_[i]) + (mean_beta_wordlen * all_wordlen_[i]) + (mean_beta_valence * all_valence_[i]) + (mean_beta_arousal * all_arousal_[i]) + (mean_beta_mlist * all_m_list_[i]) + (mean_beta_mpool * all_means_mpool_[i]) + (mean_intercept))

print(np.array(np.shape(pred)))
print(pred)

pred = np.array(pred)
ix_sorted = np.argsort(all_probs_)
final_predictions = pred[ix_sorted]

# plt.plot(range(568), np.sort(all_probs_), label = 'Actual', linewidth=5)
# plt.scatter(range(568), final_predictions, label ='Predictions', color = 'orange')
# print(np.array(final_predictions).shape)
# from scipy.stats.stats import pearsonr
# plt.xlabel("Word Number", size = 13.5)
# plt.ylabel("Recall Probability", size = 13.5)
# plt.legend()
# plt.savefig('WordPredictions_ave_scatter.pdf')
# plt.show()

plt.scatter(all_probs_, final_predictions.flatten())
plt.xlabel("Actual Recall Probability", size = 13.5)
plt.ylabel("Model Prediction", size = 13.5)
plt.savefig('ActvsFit_aver.pdf')
plt.show()



"""

"""